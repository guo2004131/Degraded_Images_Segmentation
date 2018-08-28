#!/bin/bash

source /home/dg/PycharmProjects/Enviroments/pyenv2_pt4/bin/activate

model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180827-140656/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bg1 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bg2 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bg3 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bg4 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bg5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbg1 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbg2 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbg3 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbg4 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbg5 -m $model -g 1

python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bm5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bm10 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bm15 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bm20 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds bm25 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbm5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbm10 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbm15 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbm20 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rbm25 -m $model -g 1

python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds h1.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds h2.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds h2.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds h3.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds h3.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhcap1.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhcap2.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhcap2.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhcap3.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhcap3.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhdn1.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhdn2.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhdn2.5 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhdn3.0 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rhdn3.5 -m $model -g 1

python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds nsp0.02 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds nsp0.04 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds nsp0.06 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds nsp0.08 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds nsp0.10 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rnsp0.02 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rnsp0.04 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rnsp0.06 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rnsp0.08 -m $model -g 1
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -d VOC -ds rnsp0.10 -m $model -g 1
