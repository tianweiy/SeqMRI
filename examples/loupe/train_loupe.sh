set -x

PRE_SELECT=$1   # do we preselect center measurements before future sampling, should be True for all experiments 
PRE_SELECT_NUM=$2  # number of preselect line / pixels in 1d / 2d sampling case  
LOSS_TYPE=$3  # loss used for training. Options: [l1, ssim, psnr]
DATASET_NAME=$4 # real-knee
ACC=$5  # acceleration ratio
LR=$6   # learning rate 
LR_STEP=${7}  # epoch for learning rate decay 
GAMMA=${8}   # leraning rate decay ratio 
RESOLUTION=${9}   # image resolution, default to [128, 128]
ROTATE=${10}  # rotate the k-space and corresponding image target 
DEVICE=${11}  # GPU device id 
LINE_CONSTRAINED=${12}  # use 1d sampling 
RANDOM_BASELINE=${13}   # random sampling baseline 
POISSON=${14}   # poisson sampling baseline 
SPECTRUM=${15}  # baseline
EQUISPACED=${16}  # baseline 

python examples/loupe/train_loupe.py \
    --exp-dir auto \
    --dataset-name ${DATASET_NAME} \
    --accelerations ${ACC} \
    --model LOUPE \
    --input_chans 2 \
    --line-constrained ${LINE_CONSTRAINED} \
    --batch-size 16 \
    --noise-type gaussian \
    --loss_type "${LOSS_TYPE}" \
    --save-model True  \
    --preselect ${PRE_SELECT}  \
    --preselect_num ${PRE_SELECT_NUM} \
    --lr ${LR} \
    --lr-step-size ${LR_STEP} \
    --lr-gamma ${GAMMA} \
    --resolution ${RESOLUTION} ${RESOLUTION} \
    --random_rotate ${ROTATE} \
    --device ${DEVICE} \
    --random_baseline ${RANDOM_BASELINE} \
    --poisson ${POISSON}  \
    --spectrum ${SPECTRUM}  \
    --equispaced ${EQUISPACED}
