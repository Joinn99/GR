MODES=(
    # "title"
    "sem_id"
    # "hllm"
)

SPLITS=(
    "pretrain"
    # "phase1"
    # "phase2"
)

SOURCE_DOMAINS=(
    # "Movies_and_TV"
    # "Books"
    "Video_Games"
    # "Cell_Phones_and_Accessories"
    # "Sports_and_Outdoors"
)

TARGET_DOMAINS=(
    # "Movies_and_TV"
    "Books"
    # "Video_Games"
    # "Cell_Phones_and_Accessories"
    # "Sports_and_Outdoors"
)

METHODS=(
    # "average_merging"
    # "ties_merging"
    # "mask_merging"
    "task_arithmetic"
)

GPU_ID="3"

MODEL_NAME=$(python processor/merging.py --mode ${MODES[0]} \
    --source_domain ${SOURCE_DOMAINS[0]} \
    --splits $(echo ${SPLITS[@]} | tr ' ' ' ') \
    --target_domains $(echo ${TARGET_DOMAINS[@]} | tr ' ' ' ') \
    --method ${METHODS[0]} \
    --base_model_path ${zoo}/Qwen3-0.6B \
    --hllm_class_path /home/Data/tjwei/HLLM/code | grep -oP '<<<\K[^>]+(?=>>>)')

CHECKPOINT_DIR="${data}/Common/GenRec"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}"

if [ ${MODES[0]} == "sem_id" ]; then
    export BEAM_WIDTH=50
else
    export BEAM_WIDTH=5
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python processor/generate.py \
    --model_path ${CHECKPOINT_PATH} \
    --mode ${MODES[0]} \
    --domain ${SOURCE_DOMAINS[0]} \
    --beam_width ${BEAM_WIDTH} \
    --sample_num 2000 \
    --output_name ${MODEL_NAME}


# This conditional statement checks if the merged model checkpoint directory exists and removes it
# -n "${CHECKPOINT_PATH}" checks if the CHECKPOINT_PATH variable is not empty/null
# -d "${CHECKPOINT_PATH}" checks if the path exists and is a directory
# If both conditions are true, rm -rf removes the directory and all its contents recursively
# This cleanup step removes the temporary merged model after generation is complete
if [ -n "${MODEL_NAME}" ] && [ -n "${CHECKPOINT_PATH}" ] && [ -d "${CHECKPOINT_PATH}" ]; then
    rm -rf "${CHECKPOINT_PATH}"
fi