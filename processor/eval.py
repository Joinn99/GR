import torch
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from logger import get_logger, log_with_color

def get_pop_coeff(domain, threshold="2023-01-01", half_life=90, gamma=0.2):
    threshold = datetime.strptime(threshold, "%Y-%m-%d")
    inters = pd.read_csv(f"data/dataset/amazon_{domain}.csv.gz").drop(columns=["user_id"])
    inters["timestamp"] = pd.to_datetime(inters["timestamp"], unit="ms")
    inters["timediff"] = threshold - inters["timestamp"]
    inters = inters[inters["timediff"] > timedelta(days=0)]
    inters["timediff"]  = inters["timediff"] .apply(lambda x: x.days)

    inters["coeff"] = inters["timediff"].apply(lambda x: np.exp(-(x) * (np.log(2) / half_life)))
    item_coeff = inters.groupby("item_id")["coeff"].sum()
    item_coeff = item_coeff.apply(lambda x: np.power(1 + x, gamma))
    return item_coeff

def map_history_id(eval_data, item_set):
    item_set_ids = item_set.reset_index().set_index("item_id")
    history_ids = eval_data["history"].apply(lambda x: item_set_ids.loc[x, "index"].tolist())
    return history_ids

def title_eval(domain, splits, embed_model, top_k=[10, 20, 50], beam_size=5, metric="l2", rescale=False):
    item_set_path = f"data/information/amazon_{domain}.csv.gz"
    eval_data_path = f"data/messages/amazon_{domain}_test.jsonl.gz"
    item_embeddings_path = f"data/embedding/amazon_{domain}.npy"

    # load result
    item_set = pd.read_csv(item_set_path)
    eval_data = pd.read_json(eval_data_path, lines=True)
    log_with_color(logger, "INFO", f"Loaded {len(item_set)} items and {len(eval_data)} eval data", "cyan")
    item_embeddings = np.load(item_embeddings_path)
    log_with_color(logger, "INFO", f"Loaded {item_embeddings.shape[0]} item embeddings", "cyan")
    item_embeddings = torch.from_numpy(item_embeddings)

    batch_size = 256

    all_metrics = []
    from embed import generate_embeddings
    for split in splits:
        log_with_color(logger, "INFO", f"Evaluating {split}...", "magenta")
        eval_set_path = f"data/outputs/amazon_{domain}_{split}_title.jsonl"
        eval_set = pd.read_json(eval_set_path, lines=True)
        eval_set["history_ids"] = map_history_id(eval_data, item_set)

        output_titles = eval_set["output"].explode().sort_index()
        
        eval_embeddings = generate_embeddings(
            embed_model,
            output_titles.apply(lambda x: x.strip().split("\n")[0]),
            with_description=False
        )
        eval_embeddings = eval_embeddings.reshape((eval_embeddings.shape[0] // beam_size, beam_size, -1))

        all_closest_items = []

        for i in range(0, len(eval_embeddings), batch_size):
            batch_eval_embeddings = eval_embeddings[i:i+batch_size]
            history_ids = eval_set["history_ids"].iloc[i:i+batch_size].reset_index(drop=True).explode().reset_index().to_numpy().astype(int)

            # Calculate distance between batch_eval_embeddings and item_embeddings
            distance = torch.matmul(batch_eval_embeddings, item_embeddings.T)   # [batch_size, N, I]
            if metric == "cosine":
                cosine_similarity = distance / torch.norm(batch_eval_embeddings, dim=-1, keepdim=True)
                cosine_similarity = cosine_similarity / torch.norm(item_embeddings.unsqueeze(0), dim=-1, keepdim=True).transpose(1, 2)
                cosine_similarity = torch.max(cosine_similarity, dim=1).values
                cosine_similarity[tuple(history_ids.T)] = float('-inf')
                item_rankings = torch.topk(cosine_similarity, k=max(top_k), dim=1, largest=True).indices
            else:
                distance = torch.norm(batch_eval_embeddings, dim=-1, keepdim=True) + \
                            torch.norm(item_embeddings.unsqueeze(0), dim=-1, keepdim=True).transpose(1, 2) - \
                                2 * distance
                distance = torch.min(distance, dim=1).values
                distance[tuple(history_ids.T)] = float('inf')
                if rescale:
                    item_coeff = get_pop_coeff(domain, threshold="2023-01-01", half_life=90, gamma=0.1)
                    rescale_coeff = torch.from_numpy(item_set.join(item_coeff, on="item_id", how="left").fillna(0).loc[:, "coeff"].to_numpy()).unsqueeze(0)
                else:
                    rescale_coeff = 1.
                item_rankings = torch.topk(distance / rescale_coeff, k=max(top_k), dim=1, largest=False).indices

            # Get the closest item for each eva
            for i in range(item_rankings.shape[0]):
                all_closest_items.append(item_set.iloc[item_rankings[i]]["item_id"].tolist())
        eval_set["item_rankings"] = all_closest_items

        metrics = calculate_metrics(eval_set, "item_rankings", "item_id", top_k=top_k)

        metrics.update({
            "domain": domain, "split": split, "mode": "title", 
            "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        })
        all_metrics.append(metrics)
    return all_metrics

def sem_id_eval(domain, splits, top_k=[10, 20, 50]):
    item_path = f"data/information/amazon_{domain}.csv.gz"
    sem_id_path = f"data/tokens/amazon_{domain}_index.jsonl"

    # load result
    item = pd.read_csv(item_path)
    sem_id = pd.read_json(sem_id_path, lines=True)
    log_with_color(logger, "INFO", f"Loaded {len(item)} items and {len(sem_id)} sem_ids", "cyan")

    item["sem_id"] = sem_id["sem_id"].apply(str.strip)
    item = item.set_index("item_id")

    all_metrics = []
    for split in splits:
        log_with_color(logger, "INFO", f"Evaluating {split}...", "magenta")
        result_path = f"data/outputs/amazon_{domain}_{split}_sem_id.jsonl"
        result = pd.read_json(result_path, lines=True)
        result = result.join(item.loc[:, ["sem_id"]], on="item_id", how="left")
        metrics = calculate_metrics(result, "output", "sem_id", top_k=top_k)

        metrics.update({
            "domain": domain, "split": split, "mode": "sem_id", 
            "time": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        })
        all_metrics.append(metrics)
    return all_metrics

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
        log_with_color(logger, "INFO", f"NDCG@{str(k)}: {round(ndcg, 6)}, Recall@{str(k)}: {round(recall, 6)}, MRR@{str(k)}: {round(mrr, 6)}", "red")
        metrics[f"NDCG@{str(k)}"] = ndcg
        metrics[f"Recall@{str(k)}"] = recall
        metrics[f"MRR@{str(k)}"] = mrr
    return metrics

if __name__ == "__main__":
    import argparse
    import os
    DATA_PATH = os.getenv("DATA_PATH")
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="sem_id")
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--domain", type=str, nargs='+', default=["Books"])
    parser.add_argument("--split", type=str, nargs='+', default=["pretrain", "phase1", "phase2"])
    parser.add_argument("--rescale", action="store_true")
    parser.add_argument("--top_k", type=int, nargs='+', default=[10, 20, 50])
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--embed_model_path", type=str, default=f"{DATA_PATH}/zoo/Qwen3-Embedding-8B")
    args = parser.parse_args()

    logger = get_logger(__name__)
    log_with_color(logger, "INFO", f"Evaluating {args.mode} for {', '.join(args.domain)} on {', '.join(args.split)} with top_k={', '.join(map(str, args.top_k))} and beam_size={args.beam_size}", "blue")

    if args.mode == "title":
        from embed import initialize_model
        embed_model = initialize_model(
            model_path=args.embed_model_path,
            gpu_id=args.gpu_id,
            gpu_memory_utilization=0.8,
            max_model_len=2048
        )

    all_metrics = []
    for domain in args.domain:
        if args.mode == "sem_id":
            log_with_color(logger, "INFO", f"Evaluating {args.mode} for {domain} on {args.split} with top_k={', '.join(map(str, args.top_k))}", "magenta")
            metrics = sem_id_eval(domain, args.split, top_k=args.top_k)
            all_metrics.extend(metrics)
        elif args.mode == "title":
            log_with_color(logger, "INFO", f"Evaluating {args.mode} for {domain} on {args.split} with top_k={', '.join(map(str, args.top_k))}", "magenta")
            metrics = title_eval(domain, args.split, embed_model, top_k=args.top_k, beam_size=args.beam_size, rescale=args.rescale)
            all_metrics.extend(metrics)

    output_df = pd.DataFrame(all_metrics)
    output_path = f"data/archive/amazon.csv"
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False, mode="w")
    else:
        output_df.to_csv(output_path, index=False, header=False, mode="a")
    log_with_color(logger, "INFO", f"Saved results to {output_path}", "cyan")
    log_with_color(logger, "INFO", f"Evaluation completed", "magenta")