import argparse
import pandas as pd

def get_llm(
    model_path,
):
    from vllm import LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        trust_remote_code=True
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
        input_path = f"/home/Data/tjwei/GR/data/messages/{args.domain}_test.jsonl.gz"
    else:
        input_path = f"/home/Data/tjwei/GR/data/sequences/{args.domain}_test.jsonl.gz"

    output_path = f"/home/Data/tjwei/GR/data/outputs/{args.domain}_test_{args.mode}.csv"

    df = pd.read_json(input_path, lines=True)
    llm = get_llm(args.model_path)
    outputs = batch_chat(llm, df["messages"].tolist(), args.mode)
    df["output"] = outputs
    df.drop(columns=["messages"], inplace=True)
    df.to_csv(output_path, index=False)