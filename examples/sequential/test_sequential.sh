set -x

EXP_DIR=$1
DATASET_NAME=$2

python examples/sequential/train_sequential.py \
    --exp-dir ${EXP_DIR} \
    --dataset-name ${DATASET_NAME} \
    --model SequentialSampling \
    --input_chans 2 \
    --test 