import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class FaceDataSet(Dataset):
    def __init__(self, preproc=None):
        self.size = 300
        self.img_dir = "/home/wangxin/dataset/WIDER_train/images"
        # self.img_dir = "F:/DataSets/WIDER_FACE/WIDER_train/images"
        self.preproc = preproc
        self.data = np.load("./data/train_data_widerface_filtered.npy")

    def __getitem__(self, index):
        # index =11701
        self.index = index
        img_path, target = self.data[index]
        img_path = img_path.replace("F:\\DataSets\\WIDER_FACE\\WIDER_train\\images", "/home/wangxin/dataset/WIDER_train/images").replace("\\", "/")
        img = cv2.imread(img_path)
        if img is None:
            print("img is none")
        # img, target = self.crop(img, target)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        #     img = img.transpose((1, 2, 0))
        #     for i in range(len(target)):
        #         cv2.rectangle(img, (int(target[i, 0]*300), int(target[i, 1]*300)),
        #                       (int(target[i, 2]*300), int(target[i, 3]*300)), color=(0, 0, 255), thickness=2)
        #     cv2.imwrite("img.jpg", img)
        # img = img.transpose((2, 0, 1))
        return torch.from_numpy(img), np.array(target, dtype=np.float32)

    def crop(self, img, target):
        try:
            i = np.random.randint(len(target))
            box = target[i]
            box = [int(box[i]) for i in range(len(box))]
            box_w, box_h = box[2]-box[0], box[3]-box[1]
            height, width, _ = img.shape
            if box_w >= self.size or box_h >= self.size:
                return img, target
            left, up = box[0], box[1]
            right = width - box[2]
            down = height - box[3]

            bond_box = [0, 0, 0, 0]
            if left < right:
                bond_box[0] = box[0] - np.random.randint(0, min(left, self.size - box_w)+1)
                bond_box[2] = bond_box[0] + self.size
            else:
                bond_box[2] = box[2] + np.random.randint(0, min(right, self.size - box_w)+1)
                bond_box[0] = bond_box[2] - self.size
            if up < down:
                bond_box[1] = box[1] - np.random.randint(0, min(up, self.size - box_h)+1)
                bond_box[3] = bond_box[1] + self.size
            else:
                bond_box[3] = box[3] + np.random.randint(0, min(down, self.size - box_h)+1)
                bond_box[1] = bond_box[3] - self.size
            if bond_box[0] < 0:
                bond_box[0] = 0
            if bond_box[1] < 0:
                bond_box[1] = 0
            if bond_box[2] > width:
                bond_box[2] = width
            if bond_box[3] > height:
                bond_box[3] = height
            targets_cropped = [target[i]]
            for j in range(len(target)):
                if j != i:
                    box_center = [(target[j, 0] + target[j, 2])/2 , (target[j, 1]+target[j, 3])/2]
                    if bond_box[0]<box_center[0]<bond_box[2] and bond_box[1]<box_center[1]<bond_box[3]:
                        box_selected = target[j]
                        if box_selected[0] < bond_box[0]:
                            box_selected[0] = bond_box[0]
                        if box_selected[1] < bond_box[1]:
                            box_selected[1] = bond_box[1]
                        if box_selected[2] > bond_box[2]:
                            box_selected[2] = bond_box[2]
                        if box_selected[3] > bond_box[3]:
                            box_selected[3] = bond_box[3]
                        targets_cropped.append(box_selected)
        except:
            print("error", self.index)
        return img[bond_box[1]:bond_box[3], bond_box[0]:bond_box[2]], np.array(targets_cropped) - np.array([bond_box[0], bond_box[1], bond_box[0], bond_box[1], 0])


    def __len__(self):
        return len(self.data)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)