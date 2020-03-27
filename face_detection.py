import os
import argparse
import torch
import time
import torch.utils.data
import numpy as np
from XTroch.solver import Solver
from models.facebox_tiny import FaceBoxes
from lib.dataset_ori import FaceDataSet, detection_collate, AnnotationTransform
from lib.data_agument import preproc
from lib.multibox_loss import MultiBoxLoss
from lib.prior_box import PriorBox

img_dim = 300 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order


class FaceDetection(Solver):
    def __init__(self, cfg, session=None, save=True):
        super().__init__(cfg, session=session, save=save)
        self.lr = 0.
        self.criterion = MultiBoxLoss(self.cfg.NUM_CLASS, 0.35, True, 0, True, 7, 0.35, False, self.cfg.USE_GPU)
        # self.device = torch.device('cuda:0' if self.cfg.USE_GPU else 'cpu')
        priorbox = PriorBox(self.cfg, image_size=(img_dim, img_dim))
        with torch.no_grad():
            priors = priorbox.forward()
            self.priors = priors.cuda()

    def build_model(self):
        model = FaceBoxes(phase="train", num_classes=self.cfg.NUM_CLASS)
        self.use_gpu = self.cfg.USE_GPU and torch.cuda.is_available()
        if "RESUME" in self.cfg:
            print('resume from {}'.format(self.cfg.RESUME))
            check_point = torch.load(self.cfg.RESUME)
            model.load_state_dict(check_point['model'])
        if self.use_gpu:
            model = model.cuda()
            if self.cfg.MULTI_GPU:
                model = torch.nn.DataParallel(model)
        return model

    def _create_loader(self):
        train_data = FaceDataSet(preproc=preproc(img_dim, rgb_mean), target_transform=AnnotationTransform())
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                                   num_workers=self.cfg.TRAIN.NUM_WORKERS, pin_memory=True, collate_fn=detection_collate)
        return train_loader

    def train(self):
        epoch_size = self.cfg.TRAIN.EPOCH
        for epoch in range(self.start_epoch, epoch_size):
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch)
                self.lr = self.lr_scheduler.get_lr()[0]
            loss_train = self.run_epoch(epoch, self.model, self.optim, self.data_loader, phase="train")
            if epoch > 0 and epoch % 20 == 0:
                self.logger.save_model(self.model, self.optim, self.lr_scheduler, epoch)

    def run_epoch(self, epoch, model, optim, data_loader, phase):
        if phase == 'train':
            model.train()
        else:
            model.eval()
        loss_all = 0
        Num_total = 0
        t_start = time.time()
        for batch_index, data in enumerate(data_loader):
            if self.use_gpu:
                data = [d.cuda() if not isinstance(d, list) else d for d in data]
            if phase == 'train':
                optim.zero_grad()
            img, targets = data
            batchNum = len(img)
            Num_total += batchNum
            targets = [anno.cuda() for anno in targets]
            out = model(img)
            loss_l, loss_c = self.criterion(out, self.priors, targets)
            loss = self.cfg.LOC_WEIGHT * loss_l + loss_c
            loss_all += loss.data.item()
            if phase == 'train':
                loss.backward()
                optim.step()
        avg_loss = loss_all/Num_total
        print("epoch:{0} | lr:{1:.4f} | loss_avg:{2:.4f} | time: {3:.4f}".format(epoch, self.lr, avg_loss, time.time()-t_start))
        return avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face detection solver")
    parser.add_argument(
        "--config_file",
        default='cfgs/ori.yaml',
        help="config file")
    parser.add_argument(
        "--devices",
        default=['0', '1'],
        type=str,
        nargs='*',
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.devices)
    solver = FaceDetection(args.config_file)
    solver.train()


