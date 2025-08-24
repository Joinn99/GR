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
import datetime
import copy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import pandas as pd

processor_dir = os.path.join(os.path.abspath(os.curdir), "processor")
sys.path.insert(0, processor_dir)
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

def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
    """
    Mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
    :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
    :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
    :return: torch.Tensor, masked parameters
    """
    # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
    num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

    # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
    kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
    # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
    mask = flattened_models_to_merge_param.abs() >= kth_values

    return flattened_models_to_merge_param * mask


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
            "first_dominant": 0,
            "other_dominant": 0,
            "total_params": 0,
            "r_all_same_sign": 0.0,
            "r_sign_conflict": 0.0,
            "r_all_zero": 0.0,
            "r_mixed_nonzero": 0.0,
        }

    pos = (flattened_tensor > 0)
    neg = (flattened_tensor < 0)
    zero = (flattened_tensor == 0)
    all_same_sign = (pos.all(dim=0) | neg.all(dim=0)).sum().item()                                  # param all positive or all negative
    sign_conflict = ((pos.any(dim=0) & neg.any(dim=0))).sum().item()                                # param conflict
    all_zero = zero.all(dim=0).sum().item()                                                         # param all zero
    first_dominant = ((pos[-1] | neg[-1]) & zero[:-1].all(dim=0)).sum().item()                            # param in first model dominates
    total_params = flattened_tensor.shape[1]
    other_dominant = int(total_params - all_same_sign - sign_conflict - all_zero - first_dominant)  # param in other models dominates
    

    def r(x: int) -> float:
        return (float(x) / float(total_params)) if total_params > 0 else 0.0

    return {
        "all_same_sign": int(all_same_sign),
        "sign_conflict": int(sign_conflict),
        "all_zero": int(all_zero),
        "first_dominant": int(first_dominant),
        "other_dominant": int(other_dominant),
        "total_params": int(total_params),
        "r_all_same_sign": r(all_same_sign),
        "r_sign_conflict": r(sign_conflict),
        "r_all_zero": r(all_zero),
        "r_first_dominant": r(first_dominant),
        "r_other_dominant": r(other_dominant),
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
        self.modes = args.modes
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
        self.param_value_mask_rate = args.param_value_mask_rate

    def _load_models(self):
        # Resolve model paths similarly to processor.merge.get_models usage in notebook
        if len(self.modes) > 1:
            model_names = [get_model_path(mode, self.splits[0], self.target_domains[0]) for mode in self.modes]
        elif len(self.splits) > 1:
            model_names = [get_model_path(self.modes[0], split, self.target_domains[0]) for split in self.splits]
        elif len(self.target_domains) > 1:
            model_names = [get_model_path(self.modes[0], self.splits[0], domain) for domain in self.target_domains]
        else:
            raise ValueError("No duplicate mode, split, or target domain")

        merged_model, models_to_merge = get_models(
            mode=self.modes[0],
            merged_model_name=model_names[0],
            models_to_merge_names=model_names[1:] if len(self.modes) == 1 else [],
            method="ties_merging",
            base_model_path=self.base_model_path,
            class_path=self.hllm_class_path,
        )
        if len(self.modes) > 1:
            for mode, name in zip(self.modes[1:], model_names[1:]):
                _, new_models_to_merge = get_models(
                    mode=mode,
                    merged_model_name=name,
                    models_to_merge_names=[],
                    method="average_merging",
                    base_model_path=self.base_model_path,
                    class_path=self.hllm_class_path,
                )
                models_to_merge.extend(new_models_to_merge)

        merged_model = merged_model.item_llm.cpu() if self.modes[0] == "hllm" else merged_model.model.cpu()
        for i, model in enumerate(models_to_merge):
            mode = self.modes[0] if len(self.modes) == 1 else self.modes[i]
            models_to_merge[i] = model.item_llm.cpu() if mode == "hllm" else model.model.cpu()
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
            "mode": "+".join(self.modes),
            "splits": "+".join(self.splits),
            "domains": "+".join(self.target_domains),
            "time": datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
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

        group_name = "-".join([
            "+".join(self.modes),
            "+".join(self.splits),
            "+".join([domain[:3] for domain in self.target_domains]),
        ])
        try:
            merged_model, models_to_merge = self._load_models()
            flattened = self._compute_flattened_vectors(merged_model, models_to_merge)
            if self.param_value_mask_rate > 0:
                flattened = mask_smallest_magnitude_param_values(flattened, self.param_value_mask_rate)
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
    parser.add_argument("--modes", nargs="+", default=["hllm"], help="Model mode")
    parser.add_argument("--splits", nargs="+", default=["phase2"], help="List of splits")
    parser.add_argument("--source_domain", type=str, default="Video_Games", help="Source domain name")
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
    parser.add_argument("--hist_dir", type=str, default="data/stats", help="Directory to export histogram CSVs")
    parser.add_argument("--param_value_mask_rate", type=float, default=0.0, help="Mask rate of the smallest-magnitude parameter values")

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

    args.modes = ["sem_id"]
    args.source_domain = "Video_Games"
    args.splits = ["phase2"]

    domains = ["Video_Games", "Movies_and_TV", "Sports_and_Outdoors", "Books", "Cell_Phones_and_Accessories"]
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            args.target_domains = [domains[i], domains[j]]
            args.splits = ["phase2"]
            args.hllm_class_path = "/data/tjwei/HLLM/code"
            # Ensure deterministic default dtype for any CPU ops
            torch.set_grad_enabled(False)
            StatsComputer(args).run()
