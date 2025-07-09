export DATASET="Sports_and_Outdoors"
export SPLIT="pretrain"

export ZOO_PATH=/home/Data/zoo
export MODE="title"
export OUTPUT_DIR="/home/Data/Common/GenRec/${DATASET}-${SPLIT}-${MODE}"
export BATCH_SIZE=8 # * 4 grad acc

export EPOCHS=2
export LEARNING_RATE=5e-5


mkdir -p ${OUTPUT_DIR}
envsubst <config/${MODE}.yml > ${OUTPUT_DIR}/fine_tune_config.yml

CUDA_VISIBLE_DEVICES=2 tune run full_finetune_single_device \
    --config ${OUTPUT_DIR}/fine_tune_config.yml
