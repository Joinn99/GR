# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import argparse
import logging
import os
from tqdm import tqdm
from typing import Optional

tqdm.pandas()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        return f"{color}{formatted}{reset}"


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration with colored output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create colored formatter
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)


def log_with_color(logger, level: str, message: str, color: Optional[str] = None) -> None:
    """Log a message with optional custom color.
    
    Args:
        logger: Logger instance
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Message to log
        color: Optional custom color (red, green, blue, yellow, magenta, cyan)
    """
    # Custom color mapping
    custom_colors = {
        'red': '\033[31m',
        'green': '\033[32m',
        'blue': '\033[34m',
        'yellow': '\033[33m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'reset': '\033[0m'
    }
    
    if color and color in custom_colors:
        colored_message = f"{custom_colors[color]}{message}{custom_colors['reset']}"
        getattr(logger, level.lower())(colored_message)
    else:
        getattr(logger, level.lower())(message)


def preprocess_item(item_path: str) -> pd.DataFrame:
    """Preprocess item metadata from gzipped JSON file.
    
    Args:
        item_path: Path to the gzipped JSON file containing item metadata
        
    Returns:
        DataFrame with processed item information
    """
    logger = logging.getLogger(__name__)
    log_with_color(logger, "INFO", f"Processing item metadata from: {item_path}", "cyan")
    
    df_chunks = pd.read_json(item_path, lines=True, chunksize=1000)
    df = []
    for chunk in tqdm(df_chunks, desc="Processing items", unit="k items"):
        chunk = chunk.loc[:, ['parent_asin', 'description', 'title']]
        chunk = chunk.rename(columns={'parent_asin': 'item_id'})
        chunk = chunk[chunk["title"].notna()]
        chunk = chunk[chunk["title"].apply(len) > 0]
        chunk["description"] = chunk["description"].apply(lambda x: "\n".join(x) if isinstance(x, list) else str(x))
        df.append(chunk)


    df = pd.concat(df)
    df = df.reset_index(drop=True)
    log_with_color(logger, "INFO", f"Processed {len(df)} items with valid titles", "green")
    return df


def preprocess_interaction(
    interaction_path: str,
    item_path: str,
    output_interaction_path: str,
    output_item_path: str,
    prefix: str = 'books',
    min_interactions: int = 5,
    min_date: str = '2020-01-01',
    max_item_description_len: int = 64,
    tokenizer_path: str = "Qwen/Qwen3-0.6B"
) -> None:
    """Preprocess interaction data and filter based on criteria.
    
    Args:
        interaction_path: Path to input interaction CSV file
        item_path: Path to input item metadata file
        output_interaction_path: Path to output processed interaction file
        output_item_path: Path to output processed item file
        prefix: Prefix for logging messages
        min_interactions: Minimum number of interactions required for users/items
        min_date: Minimum date filter for interactions
    """
    logger = logging.getLogger(__name__)
    log_with_color(logger, "INFO", f"Starting preprocessing for {prefix}", "magenta")
    
    # Read interaction data
    log_with_color(logger, "INFO", f"Reading interaction data from: {interaction_path}", "cyan")
    ratings = pd.read_csv(
        interaction_path,
        sep=",",
        header=0,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    
    # Read and filter items
    item = preprocess_item(item_path)
    ratings = ratings[ratings['item_id'].isin(item['item_id'])]
    
    log_with_color(logger, "INFO", f"{prefix} #data points before filter: {ratings.shape[0]}", "red")
    log_with_color(logger, "INFO", f"{prefix} #users before filter: {len(set(ratings['user_id'].values))}", "red")
    log_with_color(logger, "INFO", f"{prefix} #items before filter: {len(set(ratings['item_id'].values))}", "red")

    # Calculate interaction counts
    item_id_count = (
        ratings["item_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    
    # Apply filters
    log_with_color(logger, "INFO", f"Applying filters: min_interactions={min_interactions}, min_date={min_date}", "yellow")
    
    # Date filter
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='ms')
    ratings = ratings[ratings['timestamp'] >= min_date]
    
    # Join count information
    ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
    ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
    
    # Interaction count filters
    ratings = ratings[ratings["item_count"] >= min_interactions]
    ratings = ratings[ratings["user_count"] >= min_interactions]
    ratings = ratings.groupby('user_id').filter(lambda x: len(x['item_id']) >= min_interactions)
    
    log_with_color(logger, "INFO", f"{prefix} #data points after filter: {ratings.shape[0]}", "green")
    log_with_color(logger, "INFO", f"{prefix} #users after filter: {len(set(ratings['user_id'].values))}", "green")
    log_with_color(logger, "INFO", f"{prefix} #items after filter: {len(set(ratings['item_id'].values))}", "green")
    
    # Prepare output
    ratings = ratings[['item_id', 'user_id', 'timestamp']]
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_interaction_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_item_path), exist_ok=True)
    
    # Save processed data
    log_with_color(logger, "INFO", f"Saving processed interaction data to: {output_interaction_path}", "cyan")
    ratings['timestamp'] = ratings['timestamp'].astype('int64') // 10**6
    ratings.to_csv(output_interaction_path, index=False, header=True, compression="gzip")
    
    # Filter and save item data
    valid_items = ratings['item_id'].unique()
    item = item[item['item_id'].isin(valid_items)]
    
    if max_item_description_len > 0 and tokenizer_path is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        item.loc[:, "item_text"] = item.apply(lambda x: x.loc['title'] + x.loc['description'], axis=1)
        item.loc[:, "truncated_description"] = item.loc[:, "item_text"].progress_apply(lambda x: tokenizer.encode(x)[:max_item_description_len])
        item.loc[:, "truncated_description"] = item.loc[:, "truncated_description"].progress_apply(lambda x: tokenizer.decode(x))
        item.loc[:, "truncated_description"] = item.progress_apply(lambda x: x["truncated_description"].split(x["title"], 1)[-1], axis=1)
        item.loc[:, "description"] = item.progress_apply(lambda x: x["truncated_description"] + "..." if len(x["truncated_description"]) != len(x["description"]) else x["description"], axis=1)
        item = item.drop(columns=["item_text", "truncated_description"])

    log_with_color(logger, "INFO", f"Saving processed item data to: {output_item_path}", "cyan")
    item.to_csv(output_item_path, index=False, compression="gzip")
    
    log_with_color(logger, "INFO", f"Preprocessing completed for {prefix}", "green")


def validate_paths(file_path: str, domain: str) -> tuple[str, str, str, str]:
    """Validate and construct file paths.
    
    Args:
        file_path: Base file path
        domain: Domain name
        
    Returns:
        Tuple of (input_ratings_path, input_item_path, output_ratings_path, output_item_path)
    """
    input_ratings_path = f"{file_path}/{domain}.csv.gz"
    input_item_path = f"{file_path}/meta_{domain}.jsonl.gz"
    output_ratings_path = f"data/dataset/amazon_{domain}.csv.gz"
    output_item_path = f"data/information/amazon_{domain}.csv.gz"
    
    # Validate input files exist
    if not os.path.exists(input_ratings_path):
        raise FileNotFoundError(f"Input ratings file not found: {input_ratings_path}")
    if not os.path.exists(input_item_path):
        raise FileNotFoundError(f"Input item file not found: {input_item_path}")
    
    return input_ratings_path, input_item_path, output_ratings_path, output_item_path


def main():
    """Main function to run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon product review data for recommendation systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="/data",
        help="Base path containing the input data files"
    )
    
    parser.add_argument(
        "--domain", 
        type=str, 
        default="Cell_Phones_and_Accessories",
        help="Amazon product domain/category to process"
    )
    
    parser.add_argument(
        "--min_interactions", 
        type=int, 
        default=5,
        help="Minimum number of interactions required for users and items"
    )
    
    parser.add_argument(
        "--min_date", 
        type=str, 
        default="2020-01-01",
        help="Minimum date filter for interactions (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )

    parser.add_argument(
        "--max_item_description_len",
        type=int,
        default=64,
        help="Maximum length of item description"
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    log_with_color(logger, "INFO", "Starting Amazon data preprocessing", "magenta")
    log_with_color(logger, "INFO", f"Arguments: {vars(args)}", "blue")
    
    try:
        # Validate and construct paths
        input_ratings_path, input_item_path, output_ratings_path, output_item_path = validate_paths(
            args.file_path, args.domain
        )
        
        # Run preprocessing
        preprocess_interaction(
            interaction_path=input_ratings_path,
            item_path=input_item_path,
            output_interaction_path=output_ratings_path,
            output_item_path=output_item_path,
            prefix=args.domain,
            min_interactions=args.min_interactions,
            min_date=args.min_date,
            max_item_description_len=args.max_item_description_len,
            tokenizer_path=args.tokenizer_path
        )
        
        log_with_color(logger, "INFO", "Preprocessing completed successfully", "green")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
