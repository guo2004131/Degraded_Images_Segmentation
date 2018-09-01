#!/bin/bash

source /home/dg/PycharmProjects/Enviroments/pyenv2_pt4/bin/activate
gpu=1
#model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-1565409_TIME-20180828-003747/model_best.pth.tar
#python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg1 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-1565409_TIME-20180828-100509/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg2 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-1565409_TIME-20180828-193333/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg3 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-1565409_TIME-20180829-050130/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg4 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-1565409_TIME-20180829-150704/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg5 -m $model -d SUNRGBD -g $gpu

model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-60000_VCS-1565409_TIME-20180829-222017/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm5 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-60000_VCS-1565409_TIME-20180830-103748/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm10 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-60000_VCS-1565409_TIME-20180830-162211/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm15 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-60000_VCS-1565409_TIME-20180830-203627/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm20 -m $model -d SUNRGBD -g $gpu
model=/home/dg/PycharmProjects/Degraded_Images_Segmentation/logs/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-60000_VCS-1565409_TIME-20180831-005047/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm25 -m $model -d SUNRGBD -g $gpu


