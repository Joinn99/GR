DOMAINS=(
    "Movies_and_TV"
    "Books"
    "Video_Games"
    "Cell_Phones_and_Accessories"
    "Sports_and_Outdoors"
)

GPU_ID="3"

export MODE="sem_id"
export ZOO_PATH=${zoo}
export DATA_PATH=${data}
export CHECKPOINT_DIR="${data}/Common/GenRec"

if [ ${MODE} == "sem_id" ]; then
    export BEAM_WIDTH=50
else
    export BEAM_WIDTH=5
fi

for CUR_DOMAIN in ${DOMAINS[@]}; do
    export DOMAIN=${CUR_DOMAIN}
    for SPLIT_ID in "pretrain" "phase1" "phase2"; do
        export SPLIT=${SPLIT_ID}

        if [ ${SPLIT} == "pretrain" ]; then
            export EPOCH=2
        else
            export EPOCH=1
        fi
        
        export CHECKPOINT_PATH="${CHECKPOINT_DIR}/${DOMAIN}-${SPLIT}-${MODE}/epoch_${EPOCH}"
        # export CHECKPOINT_PATH="${ZOO_PATH}/Qwen3-0.6B"

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
            --sample_num 2000
    done
done

python processor/eval.py --mode title \
    --domain Movies_and_TV Cell_Phones_and_Accessories Books Video_Games Sports_and_Outdoors \
    --split pretrain phase1 phase2 \
    --beam_size 5 \
    --gpu_id 3 \
    --embed_model_path ${zoo}/Qwen3-Embedding-8B \
    --rescale