import argparse
import os
import shutil
import torch
import json
from typing import List

from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

from logger import get_logger, log_with_color
from merging.merging_methods import MergingMethod
from utils import get_merged_name

def hllm_from_pretrained(checkpoint_path: str = None, class_path: str = None, base_model_path: str = None):
    import sys
    import importlib
    assert class_path is not None, "class_path is required for HLLM"
    sys.path.append(class_path)
    hllm = importlib.import_module("REC.model.HLLM.hllm")

    with open(f"config/hllm.json", "r") as f:
        config = json.load(f)
    if base_model_path:
        config["item_pretrain_dir"] = base_model_path
        config["user_pretrain_dir"] = base_model_path
        config["item_llm_init"] = True
        config["user_llm_init"] = True
        config["load_pretrain"] = None
    else:
        config["load_pretrain"] = checkpoint_path
    model = hllm(config, None)
    return model

def get_model_path(mode:str, split:str, domain:str, checkpoint_dir = f"{os.environ['data']}/Common/GenRec"):
    if mode in ["title", "sem_id"]:
        epoch = "2" if split == "pretrain" else "1"
        path = f"{checkpoint_dir}/{domain}-{split}-{mode}/epoch_{epoch}"
    else:
        path = f"{checkpoint_dir}/{domain}-{split}"
    return path

def model_loader(mode:str, model_path:str = None, **kwargs):
    if mode in ["title", "sem_id"]:
        if "base_model_path" in kwargs:
            model_path = kwargs.get("base_model_path")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if mode == "sem_id" and "resize_token_embeddings" in kwargs:
                model.resize_token_embeddings(kwargs.get("resize_token_embeddings", model.vocab_size + 1024))
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path)
        return model
    else:
        return hllm_from_pretrained(model_path, **kwargs)

def get_models(mode:str,merged_model_name: str, models_to_merge_names: List[str], method: str = "average_merging", **kwargs):
    log_with_color(logger, "INFO", f"Loading merged model from: {merged_model_name}", "cyan")

    merged_model = model_loader(mode, merged_model_name, **{"hllm_class_path": kwargs.get("hllm_class_path", None)})
    models_to_merge = []
    for i, model_name in enumerate(models_to_merge_names):
        log_with_color(logger, "INFO", f"Loading model {i+1}/{len(models_to_merge_names)}: {model_name}", "cyan")
        model = model_loader(mode, model_name, **{"hllm_class_path": kwargs.get("hllm_class_path", None)})
        models_to_merge.append(model)
    
    models_to_merge.append(merged_model)
    if method not in ["average_merging", "mask_merging"]:
        if mode == "sem_id":
            kwargs["resize_token_embeddings"] = merged_model.vocab_size
        log_with_color(logger, "INFO", f"Initialize model from {kwargs.get('base_model_path', None)}", "cyan")
        merged_model = model_loader(mode, None, **kwargs)
        
    log_with_color(logger, "INFO", f"Successfully loaded {len(models_to_merge)} models", "green")
    return merged_model, models_to_merge

def save_merged_model(merged_model, merged_model_path: str, output_path: str, mode: str, name: str):
    tensor_name = "model.safetensors" if mode == "HLLM" else "model-00001-of-00001.safetensors"
    log_with_color(logger, "INFO", f"Saving merged model to: {output_path}", "blue")
    
    if not os.path.exists(output_path):
        log_with_color(logger, "INFO", f"Creating output directory: {output_path}", "yellow")
        os.makedirs(output_path, exist_ok=True)
    
    # Save the model tensors
    tensor_path = f"{output_path}/{tensor_name}"
    save_file(merged_model.to(torch.bfloat16).state_dict(), tensor_path)
    
    # Copy non-tensor files
    log_with_color(logger, "INFO", f"Copying non-tensor files from: {merged_model_path}", "cyan")
    copied_files = 0
    for file in os.listdir(merged_model_path):
        if not file.endswith(".safetensors"):
            src_path = f"{merged_model_path}/{file}"
            dst_path = f"{output_path}/{file}"
            shutil.copy(src_path, dst_path)
            copied_files += 1
    
    log_with_color(logger, "INFO", f"Copied {copied_files} non-tensor files", "green")
    log_with_color(logger, "INFO", f"Merged model name: <<<{name}>>>", "green")
    log_with_color(logger, "INFO", f"Successfully saved merged model to: {output_path}", "green")

if __name__ == "__main__":
    # Setup logger with force=True to prevent duplicate handlers
    logger = get_logger(__name__)
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="title")
    parser.add_argument("--source_domain", type=str, default="Movies_and_TV")
    parser.add_argument("--splits", nargs="+", default=["pretrain"])
    parser.add_argument("--target_domains", nargs="+", default=["Video_Games"])
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--hllm_class_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="average_merging", choices=["average_merging", "ties_merging", "mask_merging", "task_arithmetic"])
    args = parser.parse_args()
    
    log_with_color(logger, "INFO", "Starting model merging process", "magenta")
    
    # Get model paths
    output_name = get_merged_name(args.mode, args.source_domain, args.target_domains, args.splits, args.method)
    output_path = f"{os.environ['data']}/Common/GenRec/{output_name}"

    if len(args.splits) == 1:
        merged_model_path = get_model_path(args.mode, args.splits[0], args.source_domain)
        models_to_merge_paths = [
            get_model_path(args.mode, args.splits[0], target_domain)
            for target_domain in args.target_domains
        ]
    else:
        merged_model_path = get_model_path(args.mode, args.splits[0], args.source_domain)
        models_to_merge_paths = [
            get_model_path(args.mode, split, args.source_domain)
            for split in args.splits
        ]

    # Load models
    merged_model, models_to_merge = get_models(
        mode=args.mode,
        merged_model_name=merged_model_path,
        models_to_merge_names=models_to_merge_paths,
        method=args.method,
        base_model_path=args.base_model_path,
        hllm_class_path=args.hllm_class_path
    )

    # Perform merging
    merged_model = MergingMethod(args.method).get_merged_model(
        merged_model=merged_model,
        models_to_merge=models_to_merge + [merged_model],
        exclude_param_names_regex=[]
    )
    
    # Save merged model
    save_merged_model(
        merged_model=merged_model,
        merged_model_path=merged_model_path,
        output_path=output_path,
        mode=args.mode,
        name=output_name
    )

    log_with_color(logger, "INFO", "Model merging process completed", "magenta")
    

