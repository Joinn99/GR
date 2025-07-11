import pandas as pd
import numpy as np


def sem_id_eval(domain):
    result_path = f"data/outputs/amazon_{domain}_test_sem_id.csv"
    item_path = f"data/information/amazon_{domain}.csv.gz"
    sem_id_path = f"data/tokens/amazon_{domain}_index.jsonl"

    # load result
    result = pd.read_csv(result_path)
    item = pd.read_csv(item_path)
    sem_id = pd.read_json(sem_id_path, lines=True)

    result["output"] = result["output"].apply(lambda x: eval(x))
    item["sem_id"] = sem_id["sem_id"]
    item = item.set_index("item_id")
    result = result.join(item.loc[:, ["sem_id"]], on="item_id", how="left")

    matching = result.apply(lambda x: sem_id_match(x["output"], x["sem_id"]), axis=1)
    val = np.array(matching.to_list()).astype(float)
    NDCG = np.sum(val / np.log2(np.arange(2, len(val) + 2)).reshape(-1, 1), axis=1).mean()
    Recall = np.sum(val, axis=1).mean()
    print(f"NDCG: {NDCG}, Recall: {Recall}")

def sem_id_match(results, target):
    # target = target.split("<d_")[0]
    output = [target in e for e in results]
    return output
    