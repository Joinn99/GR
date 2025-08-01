# residual_balanced_kmeans.py
"""Multi‑layer Residual **Balanced K‑means** with variable cluster sizes (GPU‑only)
================================================================================
This version extends the previous implementation to support *balanced but
unequal* cluster sizes when the data cardinality *N* is **not** divisible by
*K*.  We follow the recipe

```python
q, r = divmod(N, K)
cluster_sizes = [q + 1] * r + [q] * (K - r)
```

i.e. the first *r* clusters receive **⌈N / K⌉** points and the remaining
*K − r* clusters receive **⌊N / K⌋** points, guaranteeing that
`sum(cluster_sizes) == N` and `abs(size_i − size_j) ≤ 1`.

Key changes
-----------
1. **`BalancedKMeansTorch`**
   * removes the *N mod K == 0* constraint;
   * computes `cluster_sizes` once per iteration and uses it when selecting the
     `topk` closest unassigned points for each centroid.
2. The rest of the pipeline (multi‑layer residual recursion, encoding, CLI)
   remains unchanged, as cluster balancing is encapsulated in the single‑layer
   class.

All operations stay on the configured *device* (GPU by default) to scale to
very large datasets.
"""
from __future__ import annotations

from typing import List, Sequence, Union, Optional
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

__all__ = ["BalancedKMeansTorch", "MultiResidualBalancedKMeans"]

# ---------------------------------------------------------------------------
# Single‑layer balanced K‑means (GPU‑only) with variable cluster sizes
# ---------------------------------------------------------------------------
class BalancedKMeansTorch:
    """Balanced K‑means allowing *N mod K ≠ 0* (kept on GPU).

    Parameters
    ----------
    n_clusters : int
        Number of centroids *K*.
    max_iter : int, default=100
        Maximum assignment/update iterations.
    tol : float, default=1e‑4
        Early‑stopping tolerance on centroid displacement (Frobenius norm).
    device : str | torch.device | None, default=None
        Computational device; defaults to CUDA when available.
    verbose : bool, default=False
        Whether to print per‑iteration diagnostics.
    seed : int | None, default=None
        Random seed for deterministic centroid initialisation.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Union[str, torch.device, None] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.K = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else torch.device(device)
        )
        self.verbose = verbose
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

        # Fitted parameters
        self.centroids: torch.Tensor | None = None  # (K, D)
        self.labels_: torch.Tensor | None = None    # (N,)

    # ------------------------------------------------------------------
    def _initial_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """Sample K distinct points without replacement as initial centroids."""
        N = X.size(0)
        perm = torch.randperm(N, generator=self.gen, device=self.device)
        return X[perm[: self.K]].clone()

    # ------------------------------------------------------------------
    def _cluster_sizes(self, N: int) -> List[int]:
        """Return a list of balanced (±1) cluster sizes that sum to *N*."""
        q, r = divmod(N, self.K)
        return [q + 1] * r + [q] * (self.K - r)

    # ------------------------------------------------------------------
    def fit(self, X: torch.Tensor) -> "BalancedKMeansTorch":
        X = X.to(self.device, non_blocking=True)
        if X.ndim != 2:
            raise ValueError("Input must be a 2‑D tensor of shape (N, D)")
        N, _ = X.shape
        if N < self.K:
            raise ValueError("Need at least one sample per cluster (N ≥ K)")

        centroids = self._initial_centroids(X)
        labels = torch.full((N,), -1, dtype=torch.long, device=self.device)
        sizes = self._cluster_sizes(N)  # fixed for all iterations

        for it in range(self.max_iter):
            prev_centroids = centroids.clone()
            unassigned = torch.ones(N, dtype=torch.bool, device=self.device)

            # -------------------------------- balanced assignment --------
            for k, sz in enumerate(sizes):
                idx_unassigned = torch.nonzero(unassigned, as_tuple=False).squeeze(1)
                # l2 distance^2 to centroid k
                d = ((X[idx_unassigned] - centroids[k]) ** 2).sum(dim=1)
                # pick *sz* closest points
                _, topk = torch.topk(d, k=sz, largest=False)
                chosen = idx_unassigned[topk]
                labels[chosen] = k
                unassigned[chosen] = False

                # immediate centroid update
                centroids[k] = X[labels == k].mean(dim=0)

            # -------------------------------- convergence check ----------
            shift = (centroids - prev_centroids).norm(p="fro")
            if self.verbose and it % 100 == 0:
                print(f"iter={it:02d}  shift={shift.item():.6f}")
            if shift <= self.tol:
                break

        self.centroids = centroids
        self.labels_ = labels
        return self

    # ------------------------------------------------------------------
    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# Multi‑layer residual variant (unchanged apart from doc tweaks)
# ---------------------------------------------------------------------------
class MultiResidualBalancedKMeans:
    """Recursive residual balanced K‑means supporting variable cluster sizes."""

    def __init__(
        self,
        n_clusters: Union[int, Sequence[int]],
        *,
        n_layers: Optional[int] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        device: Union[str, torch.device, None] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(n_clusters, int):
            if n_layers is None:
                raise ValueError("n_layers is required when n_clusters is an int")
            self.Ks = [int(n_clusters)] * int(n_layers)
        else:
            if n_layers is not None and n_layers != len(n_clusters):
                raise ValueError("n_layers must match len(n_clusters)")
            self.Ks = list(map(int, n_clusters))

        self.L = len(self.Ks)
        self.hyper = dict(max_iter=max_iter, tol=tol, device=device, verbose=verbose, seed=seed)
        self.verbose = verbose

        self.centroid_layers_: List[torch.Tensor] = []
        self.label_layers_: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def fit(self, X: torch.Tensor) -> "MultiResidualBalancedKMeans":
        X = X.to(self.hyper.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"), non_blocking=True)
        residual = X
        for ℓ, K in enumerate(self.Ks):
            if self.verbose:
                print(f"Layer {ℓ}  (K={K})")
            km = BalancedKMeansTorch(K, **self.hyper)
            labels = km.fit_predict(residual)
            centroids = km.centroids
            self.centroid_layers_.append(centroids)
            self.label_layers_.append(labels)
            residual = residual - centroids[labels]
        return self

    # ------------------------------------------------------------------
    def fit_predict(self, X: torch.Tensor) -> List[torch.Tensor]:
        self.fit(X)
        return self.label_layers_

    # ------------------------------------------------------------------
    def encode(self, X: torch.Tensor) -> List[torch.Tensor]:
        if not self.centroid_layers_:
            raise RuntimeError("Model not trained; call fit() first.")
        device = self.centroid_layers_[0].device
        X = X.to(device, non_blocking=True)
        residual = X
        codes: List[torch.Tensor] = []
        for centroids in self.centroid_layers_:
            d = torch.cdist(residual, centroids)
            labels = d.argmin(dim=1)
            codes.append(labels)
            residual = residual - centroids[labels]
        return codes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--cluster_sizes", type=int, nargs="+", default=[256, 256, 256])
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.from_numpy(embeddings).float().to(device)
    
    logger.info(f"Fitting model with {args.n_layers} layers and cluster sizes {args.cluster_sizes}")
    # model = MultiResidualBalancedKMeans(
    #     n_clusters=args.cluster_sizes,
    #     n_layers=args.n_layers,
    #     max_iter=1000,
    #     tol=1e-4,
    #     seed=0,
    #     verbose=True,
    # )
    # labels_layers = model.fit_predict(embeddings)

    # index = pd.DataFrame(torch.stack(labels_layers, dim=0).to("cpu").numpy().transpose(1, 0))

    from vector_quantize_pytorch import ResidualVQ

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
    index["sem_id"] = index.apply(lambda x: "".join([f"<{chr(c+97)}_{x[c]}>" for c in range(len(index.columns))]), axis=1)
    index.to_json(index_path, orient="records", lines=True)

    logger.info(f"Saving model to {model_path}")
    torch.save(residual_vq.state_dict(), model_path)
    # np.savez(
    #     f"data/tokens/amazon_{args.domain}_model.npz", 
    #     **{f"{chr(i+97)}": e.cpu().numpy() for i, e in enumerate(model.centroid_layers_)}
    # )