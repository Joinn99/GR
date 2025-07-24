
export CHECKPOINT_DIR="/home/Data/Common/GenRec"
export DOMAIN="Video_Games"
export SPLIT="pretrain"
export MODE="sem_id"
export EPOCH=1

export ZOO_PATH="/home/Data/zoo"
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
    --domain ${DOMAIN} \
    --beam_width 20