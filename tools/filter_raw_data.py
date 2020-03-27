"""
过滤WIDERFACE数据中过于小的数据。将json文件转换为txt。
"""
import cv2
import os
import numpy as np
import xml.etree.cElementTree as ET


def tranform(target):
    res = np.empty((0, 5))
    class_to_ind = {'__background': 0, 'face': 1}
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text)
            bndbox.append(cur_pt)
        area = (bndbox[3]-bndbox[1])*(bndbox[2]-bndbox[0])
        if area<900:
            # print("1")
            continue
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res = np.vstack((res, bndbox))
    return res


def filter(anno_dir):
    data = []
    img_dir = "F:\DataSets\WIDER_FACE\WIDER_train\images\\"
    for anno in os.listdir(anno_dir):
        anno_path = os.path.join(anno_dir, anno)
        target = ET.parse(anno_path).getroot()
        img_path = img_dir + anno.split("_")[0]+"--"+ anno.split("_")[1]+"\\"+ anno.replace("xml", "jpg")
        res = tranform(target)
        img = cv2.imread(img_path)
        if len(res)==0 or img is None:
            continue
        data_line = [img_path, res]
        data.append(data_line)
    data = np.array(data)
    np.save("train_data_widerface_filtered.npy", data)
        # img_path = "F:\DataSets\WIDER_FACE\WIDER_train\images" +"\\" + anno.split("_")[0]+"--"+ anno.split("_")[1]+"\\"+ anno.replace("xml", "jpg")
        # img = cv2.imread(img_path)
        # for box in res:
        #     cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 0, 255), thickness=2)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

def test_is_excited_img():
    img_dir = "F:\DataSets\WIDER_FACE\WIDER_train\images"
    data = np.load("../data/train_data_widerface_filtered.npy")
    num = 1
    for i in range(len(data)):
        img_path, target = data[i]
        img = cv2.imread(os.path.join(img_dir, img_path).replace("\\", "/"))
        if img is None:
            print(num, img_path)
            num += 1


if __name__ == '__main__':
    # filter(r"F:\DataSets\WIDER_FACE\annotations")
    data = np.load("train_data_widerface_filtered.npy")
    print(len(data))
    # test_is_excited_img()