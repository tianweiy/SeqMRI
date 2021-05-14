set -x

EXP_DIR=$1
DATASET_NAME=$2

python examples/loupe/train_loupe.py \
    --exp-dir ${EXP_DIR} \
    --dataset-name ${DATASET_NAME} \
    --model LOUPE \
    --input_chans 2 \
    --test 