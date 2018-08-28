#!/bin/bash

source /home/dg/PycharmProjects/Enviroments/pyenv2_pt4/bin/activate

model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180825-224948/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg1 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180825-233915/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg2 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-130659/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg3 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-130725/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg4 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-130809/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bg5 -m $model -d CamVid

model=
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm5 -m $model -d CamVid
model=
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm10 -m $model -d CamVid
model=
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm15 -m $model -d CamVid
model=
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm20 -m $model -d CamVid
model=
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds bm25 -m $model -d CamVid


model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-9eee8c3_TIME-20180805-172506/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds h1.5 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-8bcd71b_TIME-20180805-230253/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds h2.0 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-c6b961b_TIME-20180806-114639/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds h2.5 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-d51bbdd_TIME-20180811-190440/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds h3.0 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-d51bbdd_TIME-20180812-140049/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds h3.5 -m $model -d CamVid


model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-181031/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds nsp0.02 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-225758/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds nsp0.04 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-225835/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds nsp0.06 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180826-225913/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds nsp0.08 -m $model -d CamVid
model=/media/dg/4be9e114-dcd8-4062-9e39-880585e7fccd/results/AAAI_DegradedImages/MODEL-fcn8s-atonce_CFG-001_WEIGHT_DECAY-0.0005_LR-1e-10_INTERVAL_VALIDATE-4000_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-bcf2d18_TIME-20180827-003030/model_best.pth.tar
python /home/dg/PycharmProjects/Degraded_Images_Segmentation/test_fcn8s_atonce.py -ds nsp0.10 -m $model -d CamVid