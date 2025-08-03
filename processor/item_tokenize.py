import os
import torch
import numpy as np
import pandas as pd
import logging

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
        'RESET': '\033[0m'
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--cluster_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Configure logging with colors
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)

    embedding_path = f"data/embedding/amazon_{args.domain}.npy"
    index_path = f"data/tokens/amazon_{args.domain}_index.jsonl"
    model_path = f"data/tokens/amazon_{args.domain}_model.pth"
    
    embeddings = np.load(embedding_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.from_numpy(embeddings).float().to(device)
    
    logger.info(f"Fitting model with {args.n_layers} layers and cluster sizes {args.cluster_sizes}")

    from vector_quantize_pytorch import ResidualVQ
    torch.random.manual_seed(0)
    residual_vq = ResidualVQ(
        dim = 4096,
        num_quantizers = 3,      # specify number of quantizers
        codebook_size = 256,    # codebook size
        kmeans_init = True,   # set to True
        kmeans_iters = 10,     # number of kmeans iterations to calculate the centroids for the codebook on init
        # stochastic_sample_codes = True,
        # sample_codebook_temp = 5e-4,         # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
    ).to(device)
    quantized, indices, commit_loss = residual_vq(embeddings)
    logger.info(f"Commit loss: {commit_loss}")

    index = pd.DataFrame(indices.to("cpu").numpy())

    last_id = index.groupby(list(index.columns)).cumcount()
    logger.info(f"Saving index to {index_path}, max cluster size: {last_id.max()}")
    index[len(index.columns)] = last_id
    index = index.rename(columns={c: f"ID_{chr(c+97)}" for c in range(len(index.columns))})
    index.loc[:, "sem_id"] = index.apply(lambda x: "".join([f"<{chr(c+97)}_{x[c]}>" for c in range(len(index.columns))]), axis=1)
    index.to_json(index_path, orient="records", lines=True)

    logger.info(f"Saving model to {model_path}")
    torch.save(residual_vq.state_dict(), model_path)