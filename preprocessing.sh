DOMAINS=(
    "Video_Games"
    # "Movies_and_TV"
    # "Cell_Phones_and_Accessories"
    # "Books"
    # "Sports_and_Outdoors"
)

for DOMAIN in ${DOMAINS[@]}; do
    # python processor/preprocess.py --file_path /home/Data/dataset/Amazon --domain $DOMAIN --tokenizer_path /home/Data/zoo/Qwen3-0.6B --min_date 2017-07-01
    # python processor/embed.py --domain $DOMAIN  --model_path /home/Data/zoo/Qwen3-Embedding-8B --gpu_id 0
    # CUDA_VISIBLE_DEVICES=0 python processor/item_tokenize.py --domain $DOMAIN
    python processor/formulator.py --domain $DOMAIN --index sem_id
    python processor/item_formulator.py --domain $DOMAIN --item_group_num 5
done