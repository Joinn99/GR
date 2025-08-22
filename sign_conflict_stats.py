#!/usr/bin/env python3
"""
Compute sign-conflict statistics for merging groups, following merge_process.py's paradigm.

For each merging group (source, targets, method, split, mode):
- Load models via processor.merge.get_models (without actually merging)
- Build task vectors (finetuned - base) excluding embedding/lm_head
- Flatten to vectors, compute sign stats
- Append a stats row to data/archive/stat.tsv (TSV)
- Export per-group histogram CDFs to stats/{GROUP_NAME}.csv
"""

import os
import sys
import gc
import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

sys.path.append("./processor")
from processor.merge import get_model_path, get_models
from processor.merging.task_vector import TaskVector
from processor.utils import save_csv_with_precision, get_merged_name


DEFAULT_EXCLUDE_PARAM_NAMES_REGEX = ["embed_tokens", "lm_head"]


def task_vector_param_dict_to_single_vector(task_vector: TaskVector) -> torch.Tensor:
    """Convert TaskVector.param_dict to a single 1-D tensor with a deterministic order."""
    param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
    sorted_items = sorted(param_dict.items())
    flat_parts: List[torch.Tensor] = [p.reshape(-1).detach().to(torch.float32) for _, p in sorted_items]
    if len(flat_parts) == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


@torch.no_grad()
def compute_sign_stats(flattened_tensor: torch.Tensor) -> dict:
    """
    flattened_tensor: Tensor of shape (num_models_to_merge, num_total_params)
    Returns counts and ratios.
    """
    if flattened_tensor.numel() == 0:
        return {
            "all_same_sign": 0,
            "sign_conflict": 0,
            "all_zero": 0,
            "mixed_nonzero": 0,
            "total_params": 0,
            "r_all_same_sign": 0.0,
            "r_sign_conflict": 0.0,
            "r_all_zero": 0.0,
            "r_mixed_nonzero": 0.0,
        }

    pos_all = (flattened_tensor > 0).all(dim=0)
    neg_all = (flattened_tensor < 0).all(dim=0)
    all_same_sign = (pos_all | neg_all).sum().item()
    sign_conflict = ((flattened_tensor > 0).any(dim=0) & (flattened_tensor < 0).any(dim=0)).sum().item()
    all_zero = (flattened_tensor == 0).all(dim=0).sum().item()
    total_params = flattened_tensor.shape[1]
    mixed_nonzero = int(total_params - all_same_sign - sign_conflict - all_zero)

    def r(x: int) -> float:
        return (float(x) / float(total_params)) if total_params > 0 else 0.0

    return {
        "all_same_sign": int(all_same_sign),
        "sign_conflict": int(sign_conflict),
        "all_zero": int(all_zero),
        "mixed_nonzero": mixed_nonzero,
        "total_params": int(total_params),
        "r_all_same_sign": r(all_same_sign),
        "r_sign_conflict": r(sign_conflict),
        "r_all_zero": r(all_zero),
        "r_mixed_nonzero": r(mixed_nonzero),
    }


@torch.no_grad()
def build_hist_cdf(flattened_models_param: torch.Tensor, bins: int = 500, bin_range: Tuple[float, float] = (-0.01, 0.01)) -> pd.DataFrame:
    """
    Build cumulative histogram (CDF) per model for visualization/export.
    Returns a DataFrame with columns: bin_left, bin_right, cum_0, cum_1, ...
    """
    if flattened_models_param.numel() == 0:
        return pd.DataFrame()

    hists = [torch.histogram(flattened_models_param[i], bins=bins, range=bin_range) for i in range(flattened_models_param.shape[0])]
    # All hist share bin_edges when using same bins/range
    bin_edges = hists[0].bin_edges.detach().cpu().numpy()
    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    data = {"bin_left": bin_left, "bin_right": bin_right}
    for i, hist in enumerate(hists):
        counts = hist.hist.detach().to(torch.float32)
        denom = counts.sum().clamp(min=1.0)
        cdf = (counts.cumsum(dim=0) / denom).cpu().numpy()
        data[f"cum_{i}"] = cdf
    return pd.DataFrame(data)


class StatsComputer:
    def __init__(self, args):
        self.mode = args.mode
        self.splits = args.splits
        self.source_domain = args.source_domain
        self.target_domains = args.target_domains
        self.method = args.method
        self.base_model_path = args.base_model_path
        self.hllm_class_path = args.hllm_class_path
        self.gpu_id = args.gpu_id
        self.exclude_param_names_regex = args.exclude_param_names_regex
        self.hist_bins = args.hist_bins
        self.hist_min = args.hist_min
        self.hist_max = args.hist_max
        self.stats_out = args.stats_out
        self.hist_dir = args.hist_dir

    def _load_models(self):
        # Resolve model paths similarly to processor.merge.get_models usage in notebook
        if len(self.splits) == 1:
            merged_model_path = get_model_path(self.mode, self.splits[0], self.source_domain)
            models_to_merge_paths = [
                get_model_path(self.mode, self.splits[0], tgt) for tgt in self.target_domains
            ]
        else:
            merged_model_path = get_model_path(self.mode, self.splits[0], self.source_domain)
            models_to_merge_paths = [get_model_path(self.mode, split, self.source_domain) for split in self.splits]

        merged_model, models_to_merge = get_models(
            mode=self.mode,
            merged_model_name=merged_model_path,
            models_to_merge_names=models_to_merge_paths,
            method=self.method,
            base_model_path=self.base_model_path,
            class_path=self.hllm_class_path,
        )
        return merged_model, models_to_merge

    @torch.no_grad()
    def _compute_flattened_vectors(self, merged_model: nn.Module, models_to_merge: List[nn.Module]) -> torch.Tensor:
        task_vectors = [
            TaskVector(
                pretrained_model=merged_model,
                finetuned_model=mdl,
                exclude_param_names_regex=self.exclude_param_names_regex,
            )
            for mdl in models_to_merge
        ]
        flat_rows: List[torch.Tensor] = [task_vector_param_dict_to_single_vector(tv) for tv in task_vectors]
        if len(flat_rows) == 0:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.vstack(flat_rows)

    def _append_stat_row(self, group_name: str, stats: dict):
        os.makedirs(os.path.dirname(self.stats_out), exist_ok=True)
        row = {
            "group": group_name,
            "mode": self.mode,
            "splits": "+".join(self.splits),
            "source": self.source_domain,
            "targets": "+".join(self.target_domains),
            "method": self.method,
            **stats,
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(self.stats_out)
        save_csv_with_precision(df, self.stats_out, precision=6, index=False, header=header, mode="a")

    def _export_hist_csv(self, group_name: str, flattened_models_param: torch.Tensor):
        os.makedirs(self.hist_dir, exist_ok=True)
        df_hist = build_hist_cdf(
            flattened_models_param,
            bins=self.hist_bins,
            bin_range=(self.hist_min, self.hist_max),
        )
        out_csv = os.path.join(self.hist_dir, f"{group_name}.csv")
        # Use comma separator for histogram CSV for easier plotting in spreadsheets
        if not df_hist.empty:
            df_hist.to_csv(out_csv, index=False)

    def run(self) -> Optional[str]:
        # Optional: pin GPU id for downstream libs (though we stay on CPU)
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        group_name = get_merged_name(
            mode=self.mode,
            source_domain=self.source_domain,
            target_domains=self.target_domains,
            splits=self.splits,
            method=self.method,
        )

        merged_model, models_to_merge = self._load_models()
        try:
            flattened = self._compute_flattened_vectors(merged_model, models_to_merge)
            stats = compute_sign_stats(flattened)
            self._append_stat_row(group_name, stats)
            self._export_hist_csv(group_name, flattened)
        finally:
            # Cleanup to free memory
            del merged_model
            del models_to_merge
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        return group_name


def setup_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Sign-conflict statistics over merging groups")

    # Core parameters (mirror merge_process defaults/options)
    parser.add_argument("--mode", type=str, default="sem_id", choices=["title", "sem_id", "hllm"], help="Model mode")
    parser.add_argument("--source_domain", type=str, default="Books", help="Source domain name")
    parser.add_argument("--splits", nargs="+", default=["phase2"], help="List of splits")
    parser.add_argument("--target_domains", nargs="+", default=["Sports_and_Outdoors"], help="Target domain names")
    parser.add_argument("--method", type=str, default="ties_merging", choices=["average_merging", "ties_merging", "mask_merging", "task_arithmetic"], help="Merging method (used only to drive base model selection)")

    # Model paths
    parser.add_argument("--base_model_path", type=str, default=f"{os.getenv('zoo', '')}/Qwen3-0.6B", help="Path to base model")
    parser.add_argument("--hllm_class_path", type=str, default=None, help="Path to HLLM class (for hllm mode)")

    # Compute/loop control
    parser.add_argument("--gpu_id", type=str, default=None, help="CUDA device id (optional)")
    parser.add_argument("--exclude_param_names_regex", nargs="+", default=DEFAULT_EXCLUDE_PARAM_NAMES_REGEX, help="Param name regex to exclude")

    # Histogram export
    parser.add_argument("--hist_bins", type=int, default=500, help="Number of histogram bins")
    parser.add_argument("--hist_min", type=float, default=-0.01, help="Histogram min range")
    parser.add_argument("--hist_max", type=float, default=0.01, help="Histogram max range")

    # Outputs
    parser.add_argument("--stats_out", type=str, default="data/archive/stat.tsv", help="TSV to append stats rows")
    parser.add_argument("--hist_dir", type=str, default="stats", help="Directory to export histogram CSVs")

    # Loop selection akin to merge_process
    parser.add_argument("--loop", type=str, default="add_one_merging", choices=["all_merging", "add_one_merging", "single_test_merging"], help="Looping strategy over groups")
    return parser


def run_loop(args) -> dict:
    from test.test_loop import get_eval_groups

    # Reuse test_loop driving: for each group it will instantiate and call .run(), returning group name
    loop_fn = get_eval_groups(args.loop)
    eval_groups = loop_fn(args, StatsComputer)
    return eval_groups


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    # Ensure deterministic default dtype for any CPU ops
    torch.set_grad_enabled(False)

    # Execute across groups
    eval_groups = run_loop(args)
    # Optionally print a brief summary
    for src, entries in eval_groups.items():
        for split, group_name in entries:
            print(f"[stats] {src} / {split} -> {group_name}")


