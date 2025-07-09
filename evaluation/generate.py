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
):
    from vllm import LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,
    )
    return llm

def batch_chat(
    llm,
    messages,
    mode = "text",
):
    from vllm import SamplingParams
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=32 if mode == "text" else 4,
    )
    output = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        continue_final_message=True,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    return [e.outputs[0].text for e in output]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/Data/zoo/Qwen3-0.6B")
    parser.add_argument("--mode", type=str, default="text")
    parser.add_argument("--domain", type=str, default="Cell_Phones_and_Accessories")
    args = parser.parse_args()

    if args.mode == "text":
        input_path = f"data/messages/amazon_{args.domain}_test.jsonl.gz"
    else:
        input_path = f"data/sequences/amazon_{args.domain}_test.jsonl.gz"

    output_path = f"data/outputs/amazon_{args.domain}_test_{args.mode}.csv"

    logger.info(f"Loading data from {input_path}")
    df = pd.read_json(input_path, lines=True)
    logger.info(f"Loaded {len(df)} rows")
    llm = get_llm(args.model_path)
    prompt_prefix = "Title: " if args.mode == "text" else "Item Index: "
    df["messages"] = df["messages"].apply(lambda x: x[:-1] + [{"role": "assistant", "content": prompt_prefix}])
    logger.info(f"Batch chatting...")
    outputs = batch_chat(llm, df["messages"].tolist(), args.mode)
    logger.info(f"Batch chatting done")
    df["output"] = outputs
    df.drop(columns=["messages"], inplace=True)
    logger.info(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
