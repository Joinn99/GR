DOMAINS=(
    # "Electronics"
    # "Clothing_Shoes_and_Jewelry"
    # "Movies_and_TV"
    "Cell_Phones_and_Accessories"
)

for DOMAIN in ${DOMAINS[@]}; do
    # python processor/preprocess.py --file_path /home/Data/dataset/Amazon14 --domain $DOMAIN
    # python processor/embed.py --domain $DOMAIN
    python processor/formulator.py --domain $DOMAIN
done