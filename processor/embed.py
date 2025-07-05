import os
import argparse
import pandas as pd
import torch
import numpy as np
import logging

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
    parser = argparse.ArgumentParser(description='Generate embeddings for Amazon product data')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default="/home/Data/zoo/Qwen3-Embedding-0.6B",
        help='Path to the embedding model'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        default='Cell_Phones_and_Accessories',
        help='Domain name'
    )
    
    parser.add_argument(
        '--gpu_id',
        type=str,
        default="3",
        help='GPU device ID to use'
    )
    
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.7,
        help='GPU memory utilization ratio'
    )
    
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=1024,
        help='Maximum model length'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum token length for input processing'
    )
    
    return parser.parse_args()


def initialize_model(model_path, gpu_id, gpu_memory_utilization, max_model_len):
    """Initialize the embedding model."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from vllm import LLM
    model = LLM(
        model=model_path,
        task="embed",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    return model

def get_input(item):
    """Process input item and return formatted text for embedding."""
    instruct = "Compress the following sentence into embedding.\n"
    text = f"{instruct}Title: {item['title']}\nDescription: {item['description']}"
    return text


def generate_embeddings(model, data, max_length):
    """Generate embeddings for the input data."""
    # Process all items
    processed_inputs = data.apply(
        lambda x: get_input(x), 
        axis=1
    ).tolist()
    
    # Generate embeddings
    outputs = model.embed(processed_inputs)
    
    # Convert to tensor
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    
    return embeddings


def save_embeddings(embeddings, output_path):
    """Save embeddings to file."""
    np.save(output_path, embeddings.numpy())
    logger.info(f"Embeddings saved to {output_path}")


def main():
    """Main function to orchestrate the embedding generation process."""
    args = parse_arguments()
    
    input_csv = f"data/information/amazon_{args.domain}.csv"
    output_file = f"data/embedding/amazon_{args.domain}.npy"

    logger.info(f"Loading model from: {args.model_path}")
    model = initialize_model(
        args.model_path, 
        args.gpu_id, 
        args.gpu_memory_utilization, 
        args.max_model_len
    )
    
    logger.info(f"Loading data from: {input_csv}")
    data = pd.read_csv(input_csv)
    
    logger.info(f"Generating embeddings for {len(data)} items...")
    embeddings = generate_embeddings(model, data, args.max_length)
    
    logger.info(f"Saving embeddings to: {output_file}")
    save_embeddings(embeddings, output_file)
    
    logger.info("Embedding generation completed successfully!")


if __name__ == "__main__":
    main()