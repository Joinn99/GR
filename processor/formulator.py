import os
import random
import argparse
import pandas as pd
import logging
from tqdm import tqdm

from utils import time_split_data

from prompt import prompt_template, item_template, prediction_template

tqdm.pandas()

# Custom colored formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Get the original format
        log_message = super().format(record)
        
        # Add color based on log level
        level_name = record.levelname
        if level_name in self.COLORS:
            log_message = f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"
        
        return log_message

# Configure logging with colors
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with colored formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create colored formatter
colored_formatter = ColoredFormatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(colored_formatter)

# Add handler to logger
logger.addHandler(console_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate training data for recommendation models')
    
    parser.add_argument(
        '--domain',
        type=str,
        default='Clothing_Shoes_and_Jewelry',
        help='Domain name for the dataset'
    )
    
    parser.add_argument(
        '--max_len',
        type=int,
        default=30,
        help='Maximum length of historical items to consider'
    )
    
    parser.add_argument(
        '--min_len',
        type=int,
        default=5,
        help='Minimum length of historical items required'
    )
    
    parser.add_argument(
        '--output_format',
        type=str,
        default='json',
        choices=['json', 'csv'],
        help='Output format for the generated data'
    )

    parser.add_argument(
        '--max_user_sample',
        type=int,
        default=5,
        help='Maximum number of samples for each user'
    )

    parser.add_argument(
        '--index',
        type=str,
        default='title',
        choices=['title', 'sem_id'],
        help='Index of the item information to use'
    )
    
    return parser.parse_args()


def load_data(domain):
    """Load and preprocess the dataset."""
    logger.info(f"Loading dataset for domain: {domain}")
    
    # Load interaction data
    interaction_file = f"data/dataset/amazon_{domain}.csv.gz"
    logger.info(f"Loading interaction data from: {interaction_file}")
    df = pd.read_csv(interaction_file)
    
    # Convert timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(by="timestamp").groupby("user_id").agg(list)
    
    # Load item information
    item_file = f"data/information/amazon_{domain}.csv.gz"
    logger.info(f"Loading item information from: {item_file}")
    item = pd.read_csv(item_file)

    sem_id_file = f"data/tokens/amazon_{domain}_index.jsonl"
    if os.path.exists(sem_id_file):
        sem_id = pd.read_json(sem_id_file, lines=True)
        item["sem_id"] = sem_id["sem_id"]
    item = item.set_index("item_id").fillna("")
    
    return df, item

def formulate_message(item_info, source_list, target, index, index_prompt=None):
    if not index_prompt:
        index_prompt = index
    sep = "\n\n" if index_prompt == "title" else ","
    source_infos = item_info.loc[source_list].apply(
        lambda x: item_template[index_prompt].format(**x), axis=1
    ).tolist()
    target_info = prediction_template[index].format(**{index: item_info.loc[target][index]})
    messages = [
        {"role": "user", "content": prompt_template.format(index=index) + "\n" + sep.join(source_infos)},
        {"role": "assistant", "content": target_info}
    ]
    return messages

def assemble_user_data(
        item_id_list,
        timestamp_list,
        all_item_info,
        max_len=30,
        min_len=5,
        max_user_sample=5,
        index="title"
    ):
    """Assemble training data for a user's interaction sequence."""
    
    results = []
    item_info = all_item_info.loc[item_id_list]    

    for i in range(min_len-1, len(item_id_list)):
        idx_start = max(0, i-max_len)
        results.append(
            {
                "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], index),
                "timestamp": timestamp_list[i],
                "item_id": item_id_list[i],
                "aux": False
            }
        )
        if index == "sem_id":
            results.append(
                {
                    "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], "title", index),
                    "timestamp": timestamp_list[i],
                    "item_id": item_id_list[i],
                    "aux": True
                }
            )
            results.append(
                {
                    "messages": formulate_message(item_info, item_id_list[idx_start:i], item_id_list[i], index, "title"),
                    "timestamp": timestamp_list[i],
                    "item_id": item_id_list[i],
                    "aux": True
                }
            )

    sample_limit = max_user_sample if index == "title" else 3*max_user_sample

    if len(results) > sample_limit:
        results = random.sample(results, sample_limit)
    
    return results


def save_data(data, domain, phase, index, mode="w"):
    """Save the generated data to file."""
    if index == "title":
        output_file = f"data/messages/amazon_{domain}_{phase}.jsonl.gz"
    elif index == "sem_id":
        output_file = f"data/sequences/amazon_{domain}_{phase}.jsonl.gz"
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert to DataFrame and save as JSON
    data.to_json(output_file, orient="records", lines=True, compression="gzip", mode=mode)

    logger.info(f"Data saved to: {output_file}, {data.shape[0]} data points")


def main():
    """Main function to orchestrate the data generation process."""
    args = parse_arguments()
    
    logger.info(f"Starting data generation for domain: {args.domain}")
    
    # Load data
    df, item = load_data(args.domain)
    random.seed(0)
    # Generate training data
    output_data = df.progress_apply(
        lambda x: assemble_user_data(
            x["item_id"], x["timestamp"], item,
            max_len=args.max_len, min_len=args.min_len,
            max_user_sample=args.max_user_sample,
            index=args.index
        ), axis=1
    )
    output_data = output_data.explode()
    output_data = pd.DataFrame(
        output_data.tolist(),
        columns=["messages", "timestamp", "item_id", "aux"],
        index=output_data.index
    )

    data_split = time_split_data(output_data)
    for phase, df_phase in data_split.items():
        if phase == "test":
            df_phase = df_phase[~df_phase["aux"]]
            df_phase = df_phase.sort_values(by="timestamp").groupby(level=0).agg("last")
        else:
            if args.index == "sem_id":
                df_phase = df_phase.groupby(["user_id", "item_id", "timestamp"]).sample(n=1)
        df_phase = df_phase.reset_index().drop(columns=["aux"])
        save_data(df_phase.sort_values(by="timestamp"), args.domain, phase, args.index)
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()