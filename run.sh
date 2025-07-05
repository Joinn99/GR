export DATASET=Video_Games
export SPLIT=pretrain

export ZOO_PATH=${HOME}/zoo
export OUTPUT_DIR="${HOME}/zoo/${DATASET}-${SPLIT}"
export BATCH_SIZE=16 # * 4 grad acc

export EPOCHS=2
export LEARNING_RATE=1e-5


mkdir -p ${OUTPUT_DIR}
envsubst <config/pretrain.yml > ${OUTPUT_DIR}/fine_tune_config.yml


tune run full_finetune_single_device \
    --config ${OUTPUT_DIR}/fine_tune_config.yml
