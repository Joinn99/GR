DOMAINS=(
    "Video_Games"
    "Movies_and_TV"
    "Cell_Phones_and_Accessories"
    "Books"
)

for DOMAIN in ${DOMAINS[@]}; do
    # python processor/preprocess.py --file_path /home/Data/dataset/Amazon --domain $DOMAIN --tokenizer_path /home/Data/zoo/Qwen3-0.6B
    # CUDA_VISIBLE_DEVICES=3 python processor/embed.py --domain $DOMAIN  --model_path /home/Data/zoo/Qwen3-Embedding-0.6B
    CUDA_VISIBLE_DEVICES=3 python processor/item_tokenize.py --domain $DOMAIN
    python processor/formulator.py --domain $DOMAIN --index sem_id
done