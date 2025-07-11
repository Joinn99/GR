import pandas as pd
import numpy as np
from tqdm import tqdm
from prompt import index_template, i2t_prompt, t2i_prompt, text_template

np.random.seed(0)

def formulate_t2i_messages(target_rows: pd.DataFrame, example_rows: pd.DataFrame):
    examples = [
        text_template.format(title=row["title"], description=row["description"]) + "\n" + \
        index_template.format(index=row["sem_id"])
        for _, row in example_rows.iterrows()
    ]
    target_item_texts = [
        text_template.format(title=row["title"], description=row["description"])
        for _, row in target_rows.iterrows()
    ]
    target_item_ids = [
        index_template.format(index=row["sem_id"])
        for _, row in target_rows.iterrows()
    ]
    messages = [
        {"role": "user", "content": t2i_prompt.format(examples="\n".join(examples)) + target_item_texts[0]},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n" + target_item_ids[0]}
    ]
    for i in range(1, len(target_item_texts)):
        messages.append({"role": "user", "content": target_item_texts[i]})
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n" + target_item_ids[i]})
    return messages

def formulate_i2t_messages(target_rows: pd.DataFrame, example_rows: pd.DataFrame):
    examples = [
        index_template.format(index=row["sem_id"]) + "\n" + \
        text_template.format(title=row["title"], description=row["description"])
        for _, row in example_rows.iterrows()
    ]
    target_item_ids = [
        index_template.format(index=row["sem_id"])
        for _, row in target_rows.iterrows()
    ]
    target_item_texts = [
        text_template.format(title=row["title"], description=row["description"])
        for _, row in target_rows.iterrows()
    ]

    messages = [
        {"role": "user", "content": i2t_prompt.format(examples="\n".join(examples)) + target_item_ids[0]},
        {"role": "assistant", "content": "<think>\n\n</think>\n\n" + target_item_texts[0]}
    ]
    for i in range(1, len(target_item_ids)):
        messages.append({"role": "user", "content": target_item_ids[i]})
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n" + target_item_texts[i]})
    return messages


def construct_dataset(item: pd.DataFrame, item_group_num: int):
    t2i_messages = []
    i2t_messages = []
    for i in tqdm(range(0, item.shape[0], item_group_num)):
        item_group = item.iloc[i:i+item_group_num]
        example_rows = item.sample(2 * item_group_num)
        example_rows = example_rows[~example_rows.index.isin(item_group.index)]
        example_rows = example_rows.sample(item_group_num)
        t2i_messages.append(
            {
                "messages": formulate_t2i_messages(item_group, example_rows),
                "item_ids": ",".join(item_group["item_id"].tolist()),
                "type": "t2i"
            }
        )
        i2t_messages.append(
            {
                "messages": formulate_i2t_messages(item_group, example_rows),
                "item_ids": ",".join(item_group["item_id"].tolist()),
                "type": "i2t"
            }
        )
    return t2i_messages + i2t_messages

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Sports_and_Outdoors")
    parser.add_argument("--item_group_num", type=int, default=5)
    args = parser.parse_args()
    
    item_file_path = f"data/information/amazon_{args.domain}.csv.gz"
    sem_id_file_path = f"data/tokens/amazon_{args.domain}_index.jsonl"
    output_file_path = f"data/sequences/amazon_{args.domain}_items.jsonl.gz"

    item = pd.read_csv(item_file_path)
    sem_id = pd.read_json(sem_id_file_path, convert_dates=False, date_unit="s", lines=True)

    item["sem_id"] = sem_id["sem_id"]
    item = item.fillna("")

    dataset = pd.DataFrame(construct_dataset(item, args.item_group_num)).sample(frac=1)

    dataset.to_json(output_file_path, orient="records", lines=True)