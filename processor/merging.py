import argparse
import os
import shutil
import torch
from typing import List
from datetime import datetime

from transformers import AutoModelForCausalLM
from safetensors.torch import save_model

from logger import get_logger, log_with_color
from merging.merging_methods import MergingMethod



def get_model_path(mode:str, split:str, domain:str, checkpoint_dir = f"{os.environ['data']}/Common/GenRec"):
    if mode in ["title", "sem_id"]:
        epoch = "2" if split == "pretrain" else "1"
        path = f"{checkpoint_dir}/{domain}-{split}-{mode}/epoch_{epoch}"
    else:
        path = f"{checkpoint_dir}/{domain}-{split}"
    
    return path

def get_models(merged_model_name: str, models_to_merge_names: List[str], method: str = "average_merging"):
    log_with_color(logger, "INFO", f"Loading merged model from: {merged_model_name}", "blue")
    merged_model = AutoModelForCausalLM.from_pretrained(merged_model_name)
    
    log_with_color(logger, "INFO", f"Loading {len(models_to_merge_names)} models to merge", "blue")
    models_to_merge = []
    for i, model_name in enumerate(models_to_merge_names):
        log_with_color(logger, "INFO", f"Loading model {i+1}/{len(models_to_merge_names)}: {model_name}", "cyan")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        models_to_merge.append(model)
    
    if method in ["average_merging", "mask_merging"]:
        models_to_merge = models_to_merge + [merged_model]
    
    log_with_color(logger, "INFO", f"Successfully loaded {len(models_to_merge)} models", "red")
    return merged_model, models_to_merge

def save_merged_model(merged_model, merged_model_path: str, output_path: str, mode: str):
    tensor_name = "model.safetensors" if mode == "HLLM" else "model-00001-of-00001.safetensors"
    log_with_color(logger, "INFO", f"Saving merged model to: {output_path}", "blue")
    
    if not os.path.exists(output_path):
        log_with_color(logger, "INFO", f"Creating output directory: {output_path}", "yellow")
        os.makedirs(output_path, exist_ok=True)
    
    # Save the model tensors
    tensor_path = f"{output_path}/{tensor_name}"
    save_model(merged_model.to(torch.bfloat16), tensor_path)
    
    # Copy non-tensor files
    log_with_color(logger, "INFO", f"Copying non-tensor files from: {merged_model_path}", "cyan")
    copied_files = 0
    for file in os.listdir(merged_model_path):
        if not file.endswith(".safetensors"):
            src_path = f"{merged_model_path}/{file}"
            dst_path = f"{output_path}/{file}"
            shutil.copy(src_path, dst_path)
            copied_files += 1
            log_with_color(logger, "DEBUG", f"Copied: {file}", "cyan")
    
    log_with_color(logger, "INFO", f"Copied {copied_files} non-tensor files", "green")
    log_with_color(logger, "INFO", f"Successfully saved merged model to: {output_path}", "green")

if __name__ == "__main__":
    # Setup logger with force=True to prevent duplicate handlers
    logger = get_logger(__name__)
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    log_with_color(logger, "INFO", "Starting model merging process", "magenta")
    
    # Get model paths
    merged_model_path = get_model_path("title", "pretrain", "Movies_and_TV")
    models_to_merge_paths = [
        get_model_path("title", "pretrain", "Video_Games")
    ]

    # Load models
    merged_model, models_to_merge = get_models(merged_model_path, models_to_merge_paths, "average_merging")

    # Perform merging
    log_with_color(logger, "INFO", "Starting model merging with average_merging method", "magenta")
    merge_start_time = datetime.now()
    
    merged_model = MergingMethod("ties_merging").get_merged_model(
        merged_model=merged_model,
        models_to_merge=models_to_merge + [merged_model],
        exclude_param_names_regex=[]
    )
    
    log_with_color(logger, "INFO", "Model merging completed successfully", "green")

    # Save merged model
    output_path = f"{os.environ['data']}/Common/GenRec/merged"
    save_merged_model(merged_model, merged_model_path, output_path, "title")
    
    log_with_color(logger, "INFO", "Model merging process completed successfully", "magenta")
