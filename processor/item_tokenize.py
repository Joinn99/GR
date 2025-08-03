import os
import torch
import numpy as np
import pandas as pd
from logger import get_logger, log_with_color


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--cluster_sizes", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Configure logging with colors
    logger = get_logger(__name__)
    log_with_color(logger, "INFO", f"Starting item tokenization for {args.domain}", "magenta")

    embedding_path = f"data/embedding/amazon_{args.domain}.npy"
    index_path = f"data/tokens/amazon_{args.domain}_index.jsonl"
    model_path = f"data/tokens/amazon_{args.domain}_model.pth"

    log_with_color(logger, "INFO", f"Loading embeddings from {embedding_path}", "cyan")
    embeddings = np.load(embedding_path)
    log_with_color(logger, "INFO", f"Loaded {len(embeddings)} embeddings", "red")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.from_numpy(embeddings).float().to(device)
    
    log_with_color(logger, "INFO", f"Fitting model with {args.n_layers} layers and cluster sizes {args.cluster_sizes}", "magenta")
    try:
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
        log_with_color(logger, "INFO", f"Commit loss: {commit_loss}", "red")
    except Exception as e:
        log_with_color(logger, "ERROR", f"Item tokenization failed: {e}", "red")
        raise

    index = pd.DataFrame(indices.to("cpu").numpy())

    last_id = index.groupby(list(index.columns)).cumcount()
    log_with_color(logger, "INFO", f"Saving index to {index_path}, max cluster size: {last_id.max()}", "red")
    index[len(index.columns)] = last_id
    index = index.rename(columns={c: f"ID_{chr(c+97)}" for c in range(len(index.columns))})
    index.loc[:, "sem_id"] = index.apply(lambda x: "".join([f"<{chr(c+97)}_{x[c]}>" for c in range(len(index.columns))]), axis=1)
    index.to_json(index_path, orient="records", lines=True)
    log_with_color(logger, "INFO", f"Saved index to {index_path}", "cyan")

    log_with_color(logger, "INFO", f"Saving model to {model_path}", "cyan")
    torch.save(residual_vq.state_dict(), model_path)

    log_with_color(logger, "INFO", f"Item tokenization completed for {args.domain}", "magenta")