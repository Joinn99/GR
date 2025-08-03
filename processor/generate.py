import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from logger import get_logger, log_with_color

tqdm.pandas()

# Configure logging with colors
logger = get_logger(__name__)

def get_llm(
    model_path,
    beam_width=None,
):
    from vllm import LLM
    max_logprobs = 2 * beam_width if beam_width else 20
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
        max_logprobs=max_logprobs,
    )
    return llm

def batch_chat(
    llm,
    messages,
):
    from vllm import SamplingParams
    sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=32,
        stop=["\n"],
    )
    output = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        continue_final_message=True,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [e.outputs[0].text for e in output]

from vllm import LLM

def batch_beam_search(
    llm: LLM,
    messages,
    beam_width=20,
    max_tokens=4,
):
    from vllm.sampling_params import BeamSearchParams
    sampling_params=BeamSearchParams(
        temperature=0.0,
        beam_width=beam_width,
        max_tokens=max_tokens,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        {"prompt_token_ids": e} for e in tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True
        )
    ]
    small_batch_size = 64
    outputs = []
    for i in tqdm(range(0, len(prompts), small_batch_size)):
        batch_prompts = prompts[i:i+small_batch_size]
        batch_outputs = llm.beam_search(
            prompts=batch_prompts,
            params=sampling_params,
        )
        outputs.extend(batch_outputs)
    result_rankings = [
        [tokenizer.decode(s.tokens[-max_tokens:]) for s in output.sequences]
        for output in outputs
    ]
    return result_rankings

if __name__ == "__main__":
    import os
    DATA_PATH = os.getenv("DATA_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=f"{DATA_PATH}/zoo/Qwen3-0.6B")
    parser.add_argument("--mode", type=str, default="title")
    parser.add_argument("--split", type=str, default="phase1")
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=2000)
    args = parser.parse_args()

    np.random.seed(0)
    log_with_color(logger, "INFO", f"Generating data for {args.domain} in {args.split} with {args.mode} mode", "magenta")

    if args.mode == "title":
        input_path = f"data/messages/amazon_{args.domain}_test.jsonl.gz"
    else:
        input_path = f"data/sequences/amazon_{args.domain}_test.jsonl.gz"

    output_path = f"data/outputs/amazon_{args.domain}_{args.split}_{args.mode}.jsonl"

    log_with_color(logger, "INFO", f"Loading data from {input_path}", "cyan")
    df = pd.read_json(input_path, lines=True)
    if args.sample_num != -1:
        df = df.sample(n=args.sample_num, random_state=0).sort_index()
    log_with_color(logger, "INFO", f"Loaded {len(df)} rows", "red")
    llm = get_llm(args.model_path, beam_width=args.beam_width)
    
    prompt_prefix = "Recommended Item Title:" if args.mode == "title" else "Recommended Item Index: "
    
    log_with_color(logger, "INFO", f"Batch chatting...", "magenta")
    df["messages"] = df["messages"].apply(lambda x: x[:-1] + [{"role": "assistant", "content": prompt_prefix}])
    if args.mode == "title":
        outputs = batch_beam_search(llm, df["messages"].tolist(), beam_width=args.beam_width, max_tokens=32)
        # outputs = batch_chat(llm, df["messages"].tolist())
    else:
        outputs = batch_beam_search(llm, df["messages"].tolist(), beam_width=args.beam_width, max_tokens=4)
    log_with_color(logger, "INFO", f"Batch chatting done", "magenta")
    
    df["output"] = outputs
    df.drop(columns=["messages", "history"], inplace=True)
    log_with_color(logger, "INFO", f"Saving to {output_path}", "cyan")
    df.reset_index().to_json(output_path, lines=True, orient="records")
    
    log_with_color(logger, "INFO", f"Generation completed", "magenta")
