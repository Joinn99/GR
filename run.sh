
DOMAINS=(
    # "Video_Games"
    "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    "Sports_and_Outdoors"
)

# CUR_DOMAIN=${DOMAINS[0]}
GPU_ID="3"

for CUR_DOMAIN in ${DOMAINS[@]}; do
    for CUR_SPLIT in "phase1" "phase2"; do
        export DOMAIN=${CUR_DOMAIN}
        export SPLIT=${CUR_SPLIT}
        export ZOO_PATH=${zoo}
        export DATA_PATH=${data}

        export MODE="sem_id"

        if [ ${SPLIT} == "pretrain" ]; then
            export CHECKPOINT_PATH="${ZOO_PATH}/Qwen3-0.6B"
        elif [ ${SPLIT} == "phase1" ]; then
            export CHECKPOINT_PATH="${DATA_PATH}/Common/GenRec/${DOMAIN}-pretrain-${MODE}/epoch_2"
        elif [ ${SPLIT} == "phase2" ]; then
            export CHECKPOINT_PATH="${DATA_PATH}/Common/GenRec/${DOMAIN}-phase1-${MODE}/epoch_1"
        fi

        export OUTPUT_DIR="${DATA_PATH}/Common/GenRec/${DOMAIN}-${SPLIT}-${MODE}"
        export BATCH_SIZE=8 # * 2 export EPOCHS=3
        mkdir -p ${OUTPUT_DIR}

        if [ ${SPLIT} == "pretrain" ]; then
            export EPOCHS=3
            export LEARNING_RATE=5e-5
            envsubst <config/${MODE}.yml > ${OUTPUT_DIR}/fine_tune_config.yml
        else
            export EPOCHS=2
            export LEARNING_RATE=1e-5
            envsubst <config/${MODE}_cont.yml > ${OUTPUT_DIR}/fine_tune_config.yml
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} tune run full_finetune_single_device \
            --config ${OUTPUT_DIR}/fine_tune_config.yml
    done
done
