DOMAINS=(
    # "Video_Games"
    # "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
    "Musical_Instruments"
)

mkdir -p ${data}/dataset/Amazon

for DOMAIN in ${DOMAINS[@]}; do
    ## Check file path
    RATINGS_FILE_PATH="${data}/dataset/Amazon/${DOMAIN}.csv.gz"
    INFORMATION_FILE_PATH="${data}/dataset/Amazon/meta_${DOMAIN}.jsonl.gz"

    if [ ! -f ${RATINGS_FILE_PATH} ]; then
        echo "Error: ${RATINGS_FILE_PATH} not found, try to download..."
        wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/${DOMAIN}.csv.gz
        mv ${DOMAIN}.csv.gz ${RATINGS_FILE_PATH} || true
    fi
    if [ ! -f ${INFORMATION_FILE_PATH} ]; then
        echo "Error: ${INFORMATION_FILE_PATH} not found, try to download..."
        wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_${DOMAIN}.jsonl.gz
        mv meta_${DOMAIN}.jsonl.gz ${INFORMATION_FILE_PATH} || true
    fi

    # python processor/preprocess.py --file_path ${data}/dataset/Amazon --domain $DOMAIN --tokenizer_path ${zoo}/Qwen3-0.6B --min_date 2017-07-01
    # python processor/embed.py --domain $DOMAIN  --model_path ${zoo}/Qwen3-Embedding-8B --gpu_id 0
    python processor/item_tokenize.py --domain $DOMAIN --gpu_id 0 --n_layers 3 --cluster_sizes 256 256 256
    python processor/formulator.py --domain $DOMAIN --index title
    python processor/item_formulator.py --domain $DOMAIN --item_group_num 5
done