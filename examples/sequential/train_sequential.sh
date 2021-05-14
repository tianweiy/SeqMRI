set -x

NUM_STEP=$1 # number of sequential sampling steps (excluding preselection)
PRE_SELECT=$2 
PRE_SELECT_NUM=$3 # number of preselect line / pixels in 1d / 2d sampling case  
LOSS_TYPE=$4 # loss used for training. Options: [l1, ssim, psnr]
DATASET_NAME=$5 # real-knee
ACCELERATIONS=$6 # acceleration ratio
LR=$7 # learning rate 
DEVICE=${8} # GPU device id 
LR_STEP=${9} # epoch for learning rate decay 
GAMMA=${10}  # leraning rate decay ratio 
RESOLUTION=${11}  # image resolution, default to [128, 128]
ROTATE=${12}  # rotate the k-space and corresponding image target 
LINE_CONSTRAINED=${13}  # use 1d sampling 

python examples/sequential/train_sequential.py \
    --exp-dir auto \
    --dataset-name ${DATASET_NAME} \
    --accelerations $ACCELERATIONS\
    --model SequentialSampling \
    --input_chans 2 \
    --line-constrained ${LINE_CONSTRAINED}\
    --unet \
    --save-model True \
    --batch-size 16 \
    --noise-type gaussian \
    --loss_type "${LOSS_TYPE}" \
    --num-step ${NUM_STEP} \
    --preselect ${PRE_SELECT} \
    --preselect_num ${PRE_SELECT_NUM} \
    --lr ${LR} \
    --device ${DEVICE}  \
    --lr-step-size ${LR_STEP} \
    --lr-gamma ${GAMMA} \
    --resolution ${RESOLUTION} ${RESOLUTION} \
    --random_rotate ${ROTATE}  