import os.path as osp

import numpy as np
import PIL.Image
import scipy.misc
import scipy.io
import scipy

import glob


# To generate testing gt from the color images
files = glob.glob('/home/dg/Dropbox/Datasets/PASCALVOC/VOC_test_gt_color/*.png')
files.sort()
out_pth = '/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/Datasets/VOC/Processed_images/VOC_test_gt'
for f in files:
    lbl = PIL.Image.open(f)
    fname = f.split('/')[-1]
    lbl = np.array(lbl, dtype=np.int32)
    lbl = scipy.misc.toimage(lbl, cmin=0, cmax=255)
    out_filename = osp.join(out_pth, fname)
    scipy.misc.imsave(out_filename, lbl)


# # To generate training gt from the mat files.
# files = glob.glob('/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/Datasets/VOC/Processed_images/VOC_train_gt_mat/*.mat')
# files.sort()
# out_pth = '/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/Datasets/VOC/Processed_images/VOC_train_gt'
#
# for i, f in enumerate(files):
#     fname = f.split('/')[-1]
#     print ('%d/11355\t%s\n' % (i+1, fname))
#     mat = scipy.io.loadmat(f)
#     lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
#     lbl = scipy.misc.toimage(lbl, cmin=0, cmax=255)
#     out_filename = osp.join(out_pth, fname[:-3]+'png')
#     scipy.misc.imsave(out_filename, lbl)
