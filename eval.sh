DOMAINS=(
    "Movies_and_TV"
    "Books"
    "Video_Games"
    "Cell_Phones_and_Accessories"
    "Sports_and_Outdoors"
)

SPLITS=(
    "pretrain"
    "phase1"
    "phase2"
)

GPU_ID="1"

export MODE="sem_id"
export ZOO_PATH=${zoo}
export DATA_PATH=${data}
export CHECKPOINT_DIR="${data}/Common/GenRec"

if [ ${MODE} == "sem_id" ]; then
    export BEAM_WIDTH=50
else
    export BEAM_WIDTH=5
fi

RECOMPUTE=true

if [ ${RECOMPUTE} == true ]; then
for CUR_DOMAIN in ${DOMAINS[@]}; do
    export DOMAIN=${CUR_DOMAIN}
    for SPLIT_ID in ${SPLITS[@]}; do
        export SPLIT=${SPLIT_ID}

        if [ ${SPLIT} == "pretrain" ]; then
            export EPOCH=2
        else
            export EPOCH=1
        fi
        
        export CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DOMAIN}-${SPLIT}-${MODE}/epoch_${EPOCH}"
        # export CHECKPOINT_PATH="${CHECKPOINT_DIR}/merged"

        if [ ${MODE} == "sem_id" ]; then
        python processor/add_tokens.py \
            --checkpoint_dir ${CHECKPOINT_DIR} \
            --domain ${DOMAIN} \
            --split ${SPLIT} \
            --epoch ${EPOCH} \
            --base_model_path ${ZOO_PATH}/Qwen3-0.6B
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} python processor/generate.py \
            --model_path ${CHECKPOINT_PATH} \
            --mode ${MODE} \
            --split ${SPLIT} \
            --domain ${DOMAIN} \
            --beam_width ${BEAM_WIDTH} \
            --sample_num 10000
    done
done
fi

python processor/eval.py --mode ${MODE} \
    --domain $(echo ${DOMAINS[@]} | tr ' ' ' ') \
    --split $(echo ${SPLITS[@]} | tr ' ' ' ') \
    --beam_size ${BEAM_WIDTH} \
    --gpu_id ${GPU_ID} \
    --embed_model_path ${zoo}/Qwen3-Embedding-8B \
    --rescale