DOMAINS=(
    "Video_Games"
    "Movies_and_TV"
    "Cell_Phones_and_Accessories"
    "Books"
)

for DOMAIN in ${DOMAINS[@]}; do
    python processor/preprocess.py --file_path /data/dataset/Amazon --domain $DOMAIN --tokenizer_path /data/zoo/Qwen3-0.6B
    # python processor/embed.py --domain $DOMAIN
    python processor/formulator.py --domain $DOMAIN
done