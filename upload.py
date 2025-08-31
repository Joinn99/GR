from huggingface_hub import upload_file, upload_large_folder


for domain in [
    # "Books",
    # "Health_and_Household",
    "Movies_and_TV",
    "Video_Games",
    "Sports_and_Outdoors",
    "Software",
    "Office_Products",
    "Musical_Instruments",
    # "Cell_Phones_and_Accessories",
]:
    for split in ["pretrain", "phase1", "phase2"]:
        epoch = "2" if split == "pretrain" else "1"
        upload_file(
            path_or_fileobj=f"/home/Data/Common/GenRec/{domain}-{split}-sem_id/epoch_{epoch}/model-00001-of-00001.safetensors",
            path_in_repo=f"{domain}-{split}/model-00001-of-00001.safetensors",
            repo_id="Joinn/LC-Rec",
            repo_type="model",
        )
        upload_file(
            path_or_fileobj=f"/home/Data/Common/GenRec/{domain}-{split}-title/epoch_{epoch}/model-00001-of-00001.safetensors",
            path_in_repo=f"{domain}-{split}/model-00001-of-00001.safetensors",
            repo_id="Joinn/BIGRec",
            repo_type="model",
        )