DOMAINS=(
    # "Video_Games"
    "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
)

for CUR_DOMAIN in ${DOMAINS[@]}; do
    export DOMAIN=${CUR_DOMAIN}
    export SPLIT="phase1"
    export ZOO_PATH="/data/zoo"
    
    export MODE="sem_id"

    if [ ${SPLIT} == "pretrain" ]; then
        export CHECKPOINT_PATH="${ZOO_PATH}/Qwen3-0.6B"
    elif [ ${SPLIT} == "phase1" ]; then
        export CHECKPOINT_PATH="/data/Common/GenRec/${DOMAIN}-pretrain-${MODE}/epoch_2"
    elif [ ${SPLIT} == "phase2" ]; then
        export CHECKPOINT_PATH="/data/Common/GenRec/${DOMAIN}-phase1-${MODE}/epoch_2"
    fi

    export OUTPUT_DIR="/data/Common/GenRec/${DOMAIN}-${SPLIT}-${MODE}"
    export BATCH_SIZE=8 # * 2 export EPOCHS=3
    if [ ${SPLIT} == "pretrain" ]; then
        export EPOCHS=3
        export LEARNING_RATE=5e-5
    else
        export EPOCHS=2
        export LEARNING_RATE=1e-5
    fi

    mkdir -p ${OUTPUT_DIR}
    envsubst <config/${MODE}_cont.yml > ${OUTPUT_DIR}/fine_tune_config.yml

    CUDA_VISIBLE_DEVICES=2 tune run full_finetune_single_device \
        --config ${OUTPUT_DIR}/fine_tune_config.yml
done
