#!/usr/bin/env python3
"""
Compute the magnitude of a single model task vector and export results.

For a single model (mode, split, domain):
- Load base model and finetuned model
- Build task vector (finetuned - base) excluding embedding/lm_head
- Flatten to a single vector
- Compute magnitude statistics (L1, L2 norms, mean, std, etc.)
- Export results to CSV/TSV
"""

import os
import sys
import gc
import datetime
import copy
from typing import Optional

import torch
import torch.nn as nn
import pandas as pd

processor_dir = os.path.join(os.path.abspath(os.curdir), "processor")
sys.path.insert(0, processor_dir)
from processor.merge import get_model_path, get_models
from processor.merging.task_vector import TaskVector
from processor.utils import save_csv_with_precision


DEFAULT_EXCLUDE_PARAM_NAMES_REGEX = ["embed_tokens", "lm_head"]


def task_vector_param_dict_to_single_vector(task_vector: TaskVector) -> torch.Tensor:
    """Convert TaskVector.param_dict to a single 1-D tensor with a deterministic order."""
    param_dict = copy.deepcopy(task_vector.task_vector_param_dict)
    sorted_items = sorted(param_dict.items())
    flat_parts = [p.reshape(-1).detach().to(torch.float32) for _, p in sorted_items]
    if len(flat_parts) == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


@torch.no_grad()
def compute_magnitude_stats(flattened_vector: torch.Tensor) -> dict:
    """
    Compute magnitude statistics for a single task vector.
    
    Args:
        flattened_vector: Tensor of shape (num_params,)
    
    Returns:
        Dictionary with magnitude statistics
    """
    if flattened_vector.numel() == 0:
        return {
            "l2_norm": 0.0,
            "l1_norm": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean_abs": 0.0,
            "num_params": 0,
        }
    
    l2_norm = torch.norm(flattened_vector, p=2).item()
    l1_norm = torch.norm(flattened_vector, p=1).item()
    mean = flattened_vector.mean().item()
    std = flattened_vector.std().item()
    min_val = flattened_vector.min().item()
    max_val = flattened_vector.max().item()
    mean_abs = flattened_vector.abs().mean().item()
    num_params = int(flattened_vector.numel())
    
    return {
        "l2_norm": float(l2_norm),
        "l1_norm": float(l1_norm),
        "mean": float(mean),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
        "mean_abs": float(mean_abs),
        "num_params": int(num_params),
    }


class TaskVectorMagnitudeComputer:
    def __init__(self, args):
        self.mode = args.mode
        self.split = args.split
        self.target_domain = args.target_domain
        self.base_model_path = args.base_model_path
        self.hllm_class_path = args.hllm_class_path
        self.gpu_id = args.gpu_id
        self.exclude_param_names_regex = args.exclude_param_names_regex
        self.output_file = args.output_file
        self.export_vector = args.export_vector

    def _load_models(self):
        """Load base model and finetuned model."""
        model_name = get_model_path(self.mode, self.split, self.target_domain)
        
        # Use method != "average_merging" to get base model as merged_model
        # When method is not "average_merging", get_models loads base model as merged_model
        # and the finetuned model ends up in models_to_merge[0]
        merged_model, models_to_merge = get_models(
            mode=self.mode,
            merged_model_name=model_name,
            models_to_merge_names=[],
            method="ties_merging",  # Use non-average method to load base model
            base_model_path=self.base_model_path,
            class_path=self.hllm_class_path,
        )
        
        # Extract the actual models from the wrappers
        base_model = merged_model.item_llm.cpu() if self.mode == "hllm" else merged_model.model.cpu()
        finetuned_model = models_to_merge[0].item_llm.cpu() if self.mode == "hllm" else models_to_merge[0].model.cpu()
        
        return base_model, finetuned_model

    @torch.no_grad()
    def _compute_task_vector(self, base_model: nn.Module, finetuned_model: nn.Module) -> torch.Tensor:
        """Compute and flatten task vector."""
        task_vector = TaskVector(
            pretrained_model=base_model,
            finetuned_model=finetuned_model,
            exclude_param_names_regex=self.exclude_param_names_regex,
        )
        flattened = task_vector_param_dict_to_single_vector(task_vector)
        return flattened

    def _export_stats(self, stats: dict, group_name: str):
        """Export statistics to CSV/TSV file."""
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else ".", exist_ok=True)
        
        row = {
            "mode": self.mode,
            "split": self.split,
            "domain": self.target_domain,
            "group_name": group_name,
            "time": datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S"),
            **stats,
        }
        df = pd.DataFrame([row])
        header = not os.path.exists(self.output_file)
        save_csv_with_precision(df, self.output_file, precision=6, index=False, header=header, mode="a")

    def _export_vector(self, flattened_vector: torch.Tensor, group_name: str, output_dir: str):
        """Export the flattened vector to a file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{group_name}.pt")
        torch.save(flattened_vector.cpu(), output_path)
        print(f"Exported task vector to {output_path}")

    def run(self) -> Optional[str]:
        """Run the magnitude computation."""
        # Optional: pin GPU id for downstream libs (though we stay on CPU)
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        group_name = "-".join([
            self.mode,
            self.split,
            self.target_domain[:3] if len(self.target_domain) > 3 else self.target_domain,
        ])
        
        base_model = None
        finetuned_model = None
        
        try:
            base_model, finetuned_model = self._load_models()
            flattened = self._compute_task_vector(base_model, finetuned_model)
            stats = compute_magnitude_stats(flattened)
            
            self._export_stats(stats, group_name)
            
            if self.export_vector:
                output_dir = os.path.dirname(self.output_file) or "."
                vector_dir = os.path.join(output_dir, "task_vectors")
                self._export_vector(flattened, group_name, vector_dir)
            
            print(f"Computed magnitude for {group_name}: L2={stats['l2_norm']:.6f}, L1={stats['l1_norm']:.6f}")
            
        finally:
            # Cleanup to free memory
            if base_model is not None:
                del base_model
            if finetuned_model is not None:
                del finetuned_model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        return group_name


def setup_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Compute task vector magnitude for a single model")

    # Core parameters
    parser.add_argument("--mode", type=str, default="hllm", help="Model mode (hllm or title)")
    parser.add_argument("--split", type=str, default="phase2", help="Split name")
    parser.add_argument("--target_domain", type=str, default="Sports_and_Outdoors", help="Target domain name")
    
    # Model paths
    parser.add_argument("--base_model_path", type=str, default=f"{os.getenv('zoo', '')}/Qwen3-0.6B", help="Path to base model")
    parser.add_argument("--hllm_class_path", type=str, default=None, help="Path to HLLM class (for hllm mode)")

    # Compute/loop control
    parser.add_argument("--gpu_id", type=str, default=None, help="CUDA device id (optional)")
    parser.add_argument("--exclude_param_names_regex", nargs="+", default=DEFAULT_EXCLUDE_PARAM_NAMES_REGEX, help="Param name regex to exclude")

    # Outputs
    parser.add_argument("--output_file", type=str, default="data/archive/task_vector_magnitude.tsv", help="TSV to append stats rows")
    parser.add_argument("--export_vector", action="store_true", help="Export the flattened task vector as .pt file")

    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    # Ensure deterministic default dtype for any CPU ops
    torch.set_grad_enabled(False)
    
    args.hllm_class_path = "/home/Data/tjwei/HLLM/code"
    args.base_model_path = f"{os.getenv('zoo', '')}/Qwen3-0.6B"
    args.mode = "title"
    args.split = "phase2"
    args.target_domain = "Sports_and_Outdoors"
    args.output_file = "data/archive/task_vector_magnitude.tsv"
    args.export_vector = True

    computer = TaskVectorMagnitudeComputer(args)
    computer.run()

