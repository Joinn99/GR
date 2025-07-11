import argparse
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
        seed=0,
        max_tokens=32,
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
        temperature=0.7,
        beam_width=beam_width,
        max_tokens=max_tokens,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        {"prompt_token_ids": e} for e in tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, continue_final_message=True
        )
    ]
    outputs = llm.beam_search(
        prompts=prompts,
        params=sampling_params,
    )
    result_rankings = [
        [tokenizer.decode(s.tokens[-max_tokens:]) for s in output.sequences]
        for output in outputs
    ]
    return result_rankings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/Data/zoo/Qwen3-0.6B")
    parser.add_argument("--mode", type=str, default="title")
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    parser.add_argument("--beam_width", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "title":
        input_path = f"data/messages/amazon_{args.domain}_test.jsonl.gz"
    else:
        input_path = f"data/sequences/amazon_{args.domain}_test.jsonl.gz"

    output_path = f"data/outputs/amazon_{args.domain}_test_{args.mode}.csv"

    logger.info(f"Loading data from {input_path}")
    df = pd.read_json(input_path, lines=True)
    logger.info(f"Loaded {len(df)} rows")
    llm = get_llm(args.model_path, beam_width=args.beam_width)
    
    prompt_prefix = "Title: " if args.mode == "title" else "Item Index: "
    df["messages"] = df["messages"].apply(lambda x: x[:-1] + [{"role": "assistant", "content": prompt_prefix}])
    
    logger.info(f"Batch chatting...")
    outputs = batch_chat(llm, df["messages"].tolist()) if args.mode == "title" else batch_beam_search(llm, df["messages"].tolist(), beam_width=args.beam_width)
    logger.info(f"Batch chatting done")
    
    df["output"] = outputs
    df.drop(columns=["messages"], inplace=True)
    logger.info(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
