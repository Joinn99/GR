DOMAINS=(
    # "Movies_and_TV"
    # "Books"
    "Video_Games"
    # "Cell_Phones_and_Accessories"
    # "Sports_and_Outdoors"
)

export CHECKPOINT_DIR="/home/Data/Common/GenRec"
export DOMAIN=${DOMAINS[0]}
export MODE="title"
export ZOO_PATH="/home/Data/zoo"

if [ ${MODE} == "sem_id" ]; then
    export BEAM_WIDTH=50
else
    export BEAM_WIDTH=5
fi

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

    CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py \
        --model_path ${CHECKPOINT_PATH} \
        --mode ${MODE} \
        --split ${SPLIT} \
        --domain ${DOMAIN} \
        --beam_width ${BEAM_WIDTH} \
        --sample_num 2000
done