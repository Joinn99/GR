DOMAINS=(
    "Video_Games"
    # "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
)

for CUR_DOMAIN in ${DOMAINS[@]}; do
    export DOMAIN=${CUR_DOMAIN}
    export SPLIT="pretrain"

    export ZOO_PATH=/home/Data/zoo
    export MODE="sem_id"
    export OUTPUT_DIR="/home/Data/Common/GenRec/${DOMAIN}-${SPLIT}-${MODE}"
    export BATCH_SIZE=8 # * 2 grad acc

    export EPOCHS=5
    export LEARNING_RATE=5e-5

    mkdir -p ${OUTPUT_DIR}
    envsubst <config/${MODE}.yml > ${OUTPUT_DIR}/fine_tune_config.yml

    CUDA_VISIBLE_DEVICES=0 tune run full_finetune_single_device \
        --config ${OUTPUT_DIR}/fine_tune_config.yml
done
