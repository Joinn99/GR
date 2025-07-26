import pandas as pd
import numpy as np
import torch

from embed import initialize_model, generate_embeddings

def map_history_id(eval_data, item_set):
    item_set_ids = item_set.reset_index().set_index("item_id")
    history_ids = eval_data["history"].apply(lambda x: item_set_ids.loc[x, "index"].tolist())
    return history_ids

def title_eval(domain):
    item_set_path = f"/home/Data/tjwei/GR/data/information/amazon_{domain}.csv.gz"
    eval_set_path = f"/home/Data/tjwei/GR/data/outputs/amazon_{domain}_test_title.csv"
    eval_data_path = f"/home/Data/tjwei/GR/data/messages/amazon_{domain}_test.jsonl.gz"

    # load result
    eval_set = pd.read_csv(eval_set_path)
    item_set = pd.read_csv(item_set_path)
    eval_data = pd.read_json(eval_data_path, lines=True)
    eval_set["history_ids"] = map_history_id(eval_data, item_set)

    model = initialize_model(
        model_path="/home/Data/zoo/Qwen3-Embedding-8B",
        gpu_id="3",
        gpu_memory_utilization=0.8,
        max_model_len=2048
    )

    eval_embeddings = generate_embeddings(model, eval_set["output"].apply(str.strip), with_description=False)
    item_embeddings = generate_embeddings(model, item_set["title"], with_description=False)

    batch_size = 1024
    TOP_K = 100

    all_closest_items = []

    for i in range(0, len(eval_embeddings), batch_size):
        batch_eval_embeddings = eval_embeddings[i:i+batch_size]
        history_ids = eval_set["history_ids"].iloc[i:i+batch_size].reset_index(drop=True).explode().reset_index().to_numpy().astype(int)

        # Calculate distance between batch_eval_embeddings and item_embeddings
        distance = torch.cdist(batch_eval_embeddings, item_embeddings, p=2)
        # distance = torch.matmul(batch_eval_embeddings, item_embeddings.T)
        # cosine_similarity = distance / torch.norm(batch_eval_embeddings, dim=1, keepdim=True)
        # cosine_similarity = distance / torch.norm(item_embeddings, dim=1, keepdim=True).transpose(0, 1)

        # Assign -inf to the distance between eval and its history
        distance[tuple(history_ids.T)] = float('inf')
        item_rankings = torch.topk(distance, k=TOP_K, dim=1, largest=False).indices

        # Get the closest item for each eva
        for i in range(item_rankings.shape[0]):
            all_closest_items.append(item_set.iloc[item_rankings[i]]["item_id"].tolist())
    eval_set["item_rankings"] = all_closest_items


def sem_id_eval(domain):
    result_path = f"data/outputs/amazon_{domain}_test_sem_id.csv"
    item_path = f"data/information/amazon_{domain}.csv.gz"
    sem_id_path = f"data/tokens/amazon_{domain}_index.jsonl"

    # load result
    result = pd.read_csv(result_path)
    item = pd.read_csv(item_path)
    sem_id = pd.read_json(sem_id_path, lines=True)

    result["output"] = result["output"].apply(lambda x: eval(x))
    item["sem_id"] = sem_id["sem_id"].apply(str.strip)
    item = item.set_index("item_id")
    result = result.join(item.loc[:, ["sem_id"]], on="item_id", how="left")
    print(result.head())
    metrics = calculate_metrics(result, "output", "sem_id")
    print(metrics)

def id_match(results, target):
    output = [target in e for e in results]
    return output
    
def calculate_metrics(result_df, item_rankngs_col, target_col, top_k=[20,50,100]):
    matching = result_df.apply(lambda x: id_match(x[item_rankngs_col], x[target_col]), axis=1)
    metrics = {}
    for k in top_k:
        matching_k = matching.apply(lambda x: x[:k])
        ndcg = matching_k.apply(lambda x: np.sum(x / np.log2(np.arange(2, len(x) + 2)))).mean()
        recall = matching_k.apply(lambda x: np.sum(x)).mean()
        print(f"NDCG@{str(k)}: {ndcg}, Recall@{str(k)}: {recall}")
        metrics[f"NDCG@{str(k)}"] = ndcg
        metrics[f"Recall@{str(k)}"] = recall
    return metrics

if __name__ == "__main__":
    sem_id_eval("Video_Games")