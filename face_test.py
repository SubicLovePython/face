import os
import cv2
import argparse
import torch
import time
import torch.utils.data
import numpy as np
from XTroch.solver import Solver
from models.facebox_tiny import FaceBoxes
from lib.prior_box import PriorBox
from tools.box_utils import decode
import queue

"""
两个进程：1个用来加载数据，也就是把图片加载到队列中，resize, 去均值等操作。
另一个进程，就直接从队列中取，然后通过模型得到结果。
"""

img_dim = 300  # only 1024 is supported
rgb_mean = (104, 117, 123)  # bgr order


class FaceDetection(Solver):
    def __init__(self, cfg, session=None, save=True):
        super().__init__(cfg, session=session, save=save)
        self.rgb_mean = (104, 117, 123)
        self.size = (300, 300)
        self.scale = np.array([300, 300, 300, 300])
        self.conf_thres = 0.5
        priorbox = PriorBox(self.cfg, image_size=(img_dim, img_dim))
        with torch.no_grad():
            priors = priorbox.forward()
            self.priors = priors.cuda()

    def build_model(self):
        model = FaceBoxes(phase="test", num_classes=self.cfg.NUM_CLASS)
        self.use_gpu = self.cfg.USE_GPU and torch.cuda.is_available()
        if "RESUME" in self.cfg:
            print('resume from {}'.format(self.cfg.RESUME))
            check_point = torch.load(self.cfg.RESUME)
            model.load_state_dict(check_point['model'])
        if self.use_gpu:
            model = model.cuda()
        model.eval()
        return model

    def run(self):
        self.batchsize = 128
        mv_dir = "/home/wangxin/dataset/teachers/"
        save_dir = "./results/facebox_tiny"
        _total_time = time.time()
        total_model = 0
        total_prepro = 0
        for video_name in os.listdir(mv_dir):
            start_time = time.time()
            video_path = os.path.join(mv_dir, video_name)
            save_path = os.path.join(save_dir, video_name.replace("mp4", "txt"))
            video = cv2.VideoCapture(video_path)
            imgs = []
            results = []
            self.t_model = 0
            self.t_pipepline = 0
            self.t_prepro = 0
            while video.isOpened():
                ret, frame = video.read()
                if frame is None:
                    imgs = torch.from_numpy(np.array(imgs, dtype=np.float32).transpose((0, 3, 1, 2)))
                    results.extend(self.test_pipeline(imgs))
                    break
                frame = np.float32(frame) - self.rgb_mean
                _t_prepro = time.time()
                frame = cv2.resize(frame, self.size)
                self.t_prepro += time.time() - _t_prepro
                imgs.append(frame)
                if len(imgs) == self.batchsize:
                    imgs = torch.from_numpy(np.array(imgs, dtype=np.float32).transpose((0, 3, 1, 2)))
                    results.extend(self.test_pipeline(imgs))
                    imgs = []
            with open(save_path, "w") as sp:
                for line in results:
                    sp.write(" ".join([str(l) for l in line]))
                    sp.write("\n")
            print(video_name, "done! spent: ", time.time()-start_time, "s", "model_time: ", self.t_model, "pipe: ", self.t_pipepline,
                  "prepro", self.t_prepro)
            total_prepro += self.t_prepro
            total_model += self.t_pipepline
        print("spend total time: ", time.time()-_total_time, total_prepro, total_model)

    def test_pipeline(self, imgs):
        # model = self.model
        # model.eval()
        _t = time.time()
        if self.cfg.USE_GPU:
            imgs = imgs.cuda()
        loc, conf = self.model(imgs)
        self.t_model += time.time()-_t
        res = []
        for i in range(imgs.size()[0]):
            boxes = decode(loc[i].data.squeeze(0), self.priors.data, self.cfg.PRIOR.VARIANCE)
            boxes = boxes.cpu().numpy() * self.scale
            scores = conf[i].squeeze(0).data.cpu().numpy()[:, 1]
            order = scores.argmax()
            box, score = boxes[order], scores[order]
            if score>self.conf_thres:
                box = list(box)
                box.append(score)
                res.append(box)
            else:
                res.append([0, 0, 0, 0, 0])
        self.t_pipepline += time.time() - _t
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face detection solver")
    parser.add_argument(
        "--config_file",
        default='cfgs/ori.yaml',
        help="config file")
    parser.add_argument(
        "--devices",
        default=['0'],
        type=str,
        nargs='*',
    )
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.devices)
    solver = FaceDetection(args.config_file)
    solver.run()


