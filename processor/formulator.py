import os
import argparse
import pandas as pd
import logging
from tqdm import tqdm

from utils import time_split_data

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
    
    return parser.parse_args()


def load_data(domain):
    """Load and preprocess the dataset."""
    logger.info(f"Loading dataset for domain: {domain}")
    
    # Load interaction data
    interaction_file = f"data/dataset/amazon_{domain}.csv"
    logger.info(f"Loading interaction data from: {interaction_file}")
    df = pd.read_csv(interaction_file)
    
    # Convert timestamp and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").groupby("user_id").agg(list)
    
    # Load item information
    item_file = f"data/information/amazon_{domain}.csv"
    logger.info(f"Loading item information from: {item_file}")
    item = pd.read_csv(item_file).set_index("item_id").fillna("")
    
    return df, item


def assemble_user_data(item_id_list, timestamp_list, all_item_info, max_len=30, min_len=5):
    """Assemble training data for a user's interaction sequence."""
    
    results = []
    item_info = all_item_info.loc[item_id_list]
    
    for i in range(min_len-1, len(item_id_list)):
        idx_start = max(0, i-max_len)
        source_list, target = item_id_list[idx_start:i], item_id_list[i]
        source_infos = item_info.loc[source_list].apply(
            lambda x: item_template.format(**x), axis=1
        ).tolist()
        target_info = item_info.loc[target]["title"]
        
        messages = [
            {
                "role": "user", 
                "content": prompt.format(index="title") + "\n" + "\n".join(source_infos)
            },
            {
                "role": "assistant", 
                "content": target_info
            }
        ]
        if messages:
            results.append({"messages": messages, "timestamp": timestamp_list[i]})
    
    return results


def save_data(data, domain, phase, output_format):
    """Save the generated data to file."""
    output_file = f"data/messages/amazon_{domain}_{phase}.{output_format}"
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if output_format == 'json':
        # Convert to DataFrame and save as JSON
        data.to_json(output_file+"l.gz", orient="records", lines=True, compression="gzip")
    elif output_format == 'csv':
        # Convert to DataFrame and save as CSV
        data.to_csv(output_file, index=False)
    
    logger.info(f"Data saved to: {output_file}")


def main():
    """Main function to orchestrate the data generation process."""
    args = parse_arguments()
    
    # Define templates
    global prompt, item_template, prediction_template
    
    prompt = """Given the user's historical interactive items arranged in chronological order, please recommend a suitable item for the user. Please output the item {index}.
User's historical interactive items: 
"""
    item_template = """Title: {title}
Description: {description}"""
    prediction_template = """{title}"""
    
    logger.info(f"Starting data generation for domain: {args.domain}")
    
    # Load data
    df, item = load_data(args.domain)
    
    # Generate training data
    new_data = df.progress_apply(lambda x: assemble_user_data(x["item_id"], x["timestamp"], item, max_len=args.max_len, min_len=args.min_len), axis=1)
    new_data = pd.DataFrame(new_data.explode().tolist(), columns=["messages", "timestamp"])
    # Save data
    data_split = time_split_data(new_data)
    for phase, df_phase in data_split.items():
        save_data(df_phase, args.domain, phase, args.output_format)
    
    logger.info("Data generation completed successfully!")


if __name__ == "__main__":
    main()