import pandas as pd
import numpy as np
import torch
from datetime import datetime, timezone, timedelta

from embed import initialize_model, generate_embeddings

def map_history_id(eval_data, item_set):
    item_set_ids = item_set.reset_index().set_index("item_id")
    history_ids = eval_data["history"].apply(lambda x: item_set_ids.loc[x, "index"].tolist())
    return history_ids

def title_eval(domain, split, embedding_model_path, gpu_id="0", beam_size=5):
    item_set_path = f"data/information/amazon_{domain}.csv.gz"
    eval_set_path = f"data/outputs/amazon_{domain}_{split}_title.jsonl"
    eval_data_path = f"data/messages/amazon_{domain}_test.jsonl.gz"
    item_embeddings_path = f"data/embeddings/amazon_{domain}.npy"

    # load result
    eval_set = pd.read_json(eval_set_path, lines=True)
    item_set = pd.read_csv(item_set_path)
    eval_data = pd.read_json(eval_data_path, lines=True)
    eval_set["history_ids"] = map_history_id(eval_data, item_set)

    model = initialize_model(
        model_path="/home/Data/zoo/Qwen3-Embedding-8B",
        gpu_id="3",
        gpu_memory_utilization=0.8,
        max_model_len=2048
    )
    output_titles = eval_set["output"].explode().sort_index()
    eval_embeddings = generate_embeddings(
        model,
        output_titles.apply(lambda x: x.strip().split("\n")[0]),
        with_description=False
    )
    eval_embeddings = eval_embeddings.reshape((eval_embeddings.shape[0] // beam_size, beam_size, -1))
    item_embeddings = np.load(item_embeddings_path)
    item_embeddings = torch.from_numpy(item_embeddings)

    batch_size = 1024
    TOP_K = 50

    all_closest_items = []

    for i in range(0, len(eval_embeddings), batch_size):
        batch_eval_embeddings = eval_embeddings[i:i+batch_size]
        history_ids = eval_set["history_ids"].iloc[i:i+batch_size].reset_index(drop=True).explode().reset_index().to_numpy().astype(int)

        # Calculate distance between batch_eval_embeddings and item_embeddings
        distance = torch.matmul(batch_eval_embeddings, item_embeddings.T)   # [batch_size, N, I]
        cosine_similarity = distance / torch.norm(batch_eval_embeddings, dim=-1, keepdim=True)
        cosine_similarity = cosine_similarity / torch.norm(item_embeddings.unsqueeze(0), dim=-1, keepdim=True).transpose(1, 2)
        cosine_similarity = torch.max(cosine_similarity, dim=1).values

        # Assign -inf to the distance between eval and its history
        cosine_similarity[tuple(history_ids.T)] = float('-inf')
        item_rankings = torch.topk(cosine_similarity, k=TOP_K, dim=1, largest=True).indices

        # Get the closest item for each eva
        all_closest_items.append(item_set.iloc[item_rankings[i]]["item_id"].tolist())
    eval_set["item_rankings"] = all_closest_items


def sem_id_eval(domain, split):
    result_path = f"data/outputs/amazon_{domain}_{split}_sem_id.jsonl"
    item_path = f"data/information/amazon_{domain}.csv.gz"
    sem_id_path = f"data/tokens/amazon_{domain}_index.jsonl"

    # load result
    result = pd.read_json(result_path, lines=True)
    item = pd.read_csv(item_path)
    sem_id = pd.read_json(sem_id_path, lines=True)

    item["sem_id"] = sem_id["sem_id"].apply(str.strip)
    item = item.set_index("item_id")
    result = result.join(item.loc[:, ["sem_id"]], on="item_id", how="left")
    metrics = calculate_metrics(result, "output", "sem_id")
    metrics.update({
        "domain": domain, "split": split, "mode": "sem_id", 
        "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
    })
    return metrics

def id_match(results, target):
    output = [target in e for e in results]
    return output
    
def calculate_metrics(result_df, item_rankngs_col, target_col, top_k=[10,20,50]):
    matching = result_df.apply(lambda x: id_match(x[item_rankngs_col], x[target_col]), axis=1)
    metrics = {}
    for k in top_k:
        matching_k = matching.apply(lambda x: x[:k])
        ndcg = matching_k.apply(lambda x: np.sum(x / np.log2(np.arange(2, len(x) + 2)))).mean()
        recall = matching_k.apply(lambda x: np.sum(x)).mean()
        mrr = matching_k.apply(lambda x: np.sum(1 / (np.arange(1, len(x) + 1)) * x)).mean()
        print(f"NDCG@{str(k)}: {round(ndcg, 6)}, Recall@{str(k)}: {round(recall, 6)}, MRR@{str(k)}: {round(mrr, 6)}")
        metrics[f"NDCG@{str(k)}"] = ndcg
        metrics[f"Recall@{str(k)}"] = recall
        metrics[f"MRR@{str(k)}"] = mrr
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Books")
    parser.add_argument("--mode", type=str, default="sem_id")
    args = parser.parse_args()

    for split in ["pretrain", "phase1", "phase2"]:
        if args.mode == "sem_id":
            metrics = sem_id_eval(args.domain, split)
        elif args.mode == "title":
            metrics = title_eval(args.domain, split, embedding_model_path="/home/Data/zoo/Qwen3-Embedding-8B", gpu_id="0", beam_size=5)
