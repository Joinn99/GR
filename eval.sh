export DATASET="Video_Games"
export SPLIT="pretrain"
export MODE="sem_id"

export ZOO_PATH=/home/Data/zoo
export OUTPUT_DIR="/home/Data/zoo/${DATASET}-${SPLIT}-${MODE}/epoch_2"

CUDA_VISIBLE_DEVICES=0 python evaluation/generate.py \
    --model_path ${OUTPUT_DIR} \
    --mode ${MODE} \
    --domain ${DATASET}