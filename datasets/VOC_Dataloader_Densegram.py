#!/usr/bin/env python

import torch
import random
import PIL.Image
import collections
import numpy as np
import os.path as osp
from torch.utils import data
from fcn.utils import label2rgb
import matplotlib.pyplot as plt


class VOCSegDG(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    class_weights = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    class_ignore = -1

    def __init__(self, root, split='train', dataset='o', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.datasets = collections.defaultdict()
        # class 0 is the background
        self.n_classes = 21

        self.datasets['o'] = osp.join(self.root, 'Original_Images')

        self.datasets['bg1'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_1')
        self.datasets['bg2'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_2')
        self.datasets['bg3'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_3')
        self.datasets['bg4'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_4')
        self.datasets['bg5'] = osp.join(self.root, 'Degraded_Images', 'Blur_Gaussian', 'degraded_parameter_5')

        self.datasets['bm10'] = osp.join(self.root, 'Degraded_Images', 'Blur_Motion', 'degraded_parameter_10')
        self.datasets['bm20'] = osp.join(self.root, 'Degraded_Images', 'Blur_Motion', 'degraded_parameter_20')
        self.datasets['hi1'] = osp.join(self.root, 'Degraded_Images', 'Haze_I', 'degraded_parameter_1')
        self.datasets['ho1'] = osp.join(self.root, 'Degraded_Images', 'Haze_O', 'degraded_parameter_1')
        self.datasets['ns0.02'] = osp.join(self.root, 'Degraded_Images', 'Noise_Speckle', 'degraded_parameter_0.02')
        self.datasets['nsp0.02'] = osp.join(self.root, 'Degraded_Images', 'Noise_Salt_Pepper', 'degraded_parameter_0.02')

        img_dataset_dir_o = osp.join(self.root, self.datasets['o'])
        img_dataset_dir_d = osp.join(self.root, self.datasets[dataset])

        self.files = collections.defaultdict(list)

        for split in ['train', 'val']:
            imgsets_file = osp.join(root, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_o_file = osp.join(img_dataset_dir_o, 'VOC_train_images/%s.jpg' % did)
                img_d_file = osp.join(img_dataset_dir_d, 'VOC_train_images/%s.jpg' % did)
                lbl_file = osp.join(root, 'VOC_train_gt/%s.png' % did)
                self.files[split].append({
                    'img_o': img_o_file,
                    'img_d': img_d_file,
                    'lbl': lbl_file,
                })
        imgsets_file = osp.join(root, 'test.txt')
        for did in open(imgsets_file):
            did = did.strip()
            img_o_file = osp.join(img_dataset_dir_o, 'VOC_test_images/%s.jpg' % did)
            img_d_file = osp.join(img_dataset_dir_d, 'VOC_test_images/%s.jpg' % did)
            lbl_file = osp.join(root, 'VOC_test_gt/%s.png' % did)
            self.files['test'].append({
                'img_o': img_o_file,
                'img_d': img_d_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_o_file = data_file['img_o']
        img_o = PIL.Image.open(img_o_file)
        img_o = np.array(img_o, dtype=np.uint8)
        img_d_file = data_file['img_d']
        img_d = PIL.Image.open(img_d_file)
        img_d = np.array(img_d, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img_o, img_d, lbl)
        else:
            return img_o, img_d, lbl

    def transform(self, img_o, img_d, lbl):
        random_crop = False
        if random_crop:
            size = (np.array(lbl.shape) * 0.8).astype(np.uint32)
            img_o, img_d, lbl = self.random_crop(img_o, img_d, lbl, size)
        random_flip = False
        if random_flip:
            img_o, img_d, lbl = self.random_flip(img_o, img_d, lbl)

        img_o = img_o[:, :, ::-1]  # RGB -> BGR
        img_o = img_o.astype(np.float64)
        img_o -= self.mean_bgr
        img_o = img_o.transpose(2, 0, 1)
        img_o = torch.from_numpy(img_o).float()

        img_d = img_d[:, :, ::-1]  # RGB -> BGR
        img_d = img_d.astype(np.float64)
        img_d -= self.mean_bgr
        img_d = img_d.transpose(2, 0, 1)
        img_d = torch.from_numpy(img_d).float()

        lbl = torch.from_numpy(lbl).long()
        return img_o, img_d, lbl

    def untransform(self, img_o, img_d, lbl):
        img_o = img_o.numpy()
        img_o = img_o.transpose(1, 2, 0)
        img_o += self.mean_bgr
        img_o = img_o.astype(np.uint8)
        img_o = img_o[:, :, ::-1]

        img_d = img_d.numpy()
        img_d = img_d.transpose(1, 2, 0)
        img_d += self.mean_bgr
        img_d = img_d.astype(np.uint8)
        img_d = img_d[:, :, ::-1]
        # convert to color images
        # lbl = self.label_to_color_image(lbl)
        lbl[lbl == 255] = -1
        if not isinstance(lbl, np.ndarray):
            lbl = np.array(lbl)
        lbl = lbl.astype(np.int8)
        return img_o, img_d, lbl

    def label_to_color_image(self, lbl):
        return label2rgb(lbl)

    def random_crop(self, img_o, img_d, lbl, size):
        h, w = lbl.shape
        th, tw = size
        if w == tw and h == th:
            return img_o, img_d, lbl
        x1 = random.randint(0, w-tw)
        y1 = random.randint(0, h-th)
        img_o = img_o[y1:y1+th, x1:x1+tw, :]
        img_d = img_d[y1:y1+th, x1:x1+tw, :]
        lbl = lbl[y1:y1+th, x1:x1+tw]
        return img_o, img_d, lbl

    def random_flip(self, img_o, img_d, lbl):
        if random.random() < 0.5:
            return np.flip(img_o, 1).copy(), np.flip(img_d, 1).copy(), np.flip(lbl, 1).copy()
        return img_o, img_d, lbl


# For code testing
if __name__ == "__main__":
    root = '/home/dg/Dropbox/Datasets/VOC'
    dataset = VOCSegDG(root, split='test', dataset='bg2', transform=True)
    img_o, img_d, lbl = dataset.__getitem__(11)
    img_o, img_d, lbl = dataset.untransform(img_o, img_d, lbl)
    plt.subplot(221)
    plt.imshow(img_o)
    plt.subplot(222)
    plt.imshow(img_d)
    plt.subplot(223)
    plt.imshow(lbl)
    plt.show()
