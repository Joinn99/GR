import json

def add_token(level=4, codebook_size=256, start_id=151669):
    added_tokens = []
    for i in range(level):
        for j in range(codebook_size):
            added_tokens.append({
                "id": start_id + i * codebook_size + j,
                "content": f"<{chr(i+97)}_{j}>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True
            },)
    return added_tokens

if __name__ == "__main__":
    tok = json.load(open("/home/Data/zoo/Qwen3-0.6B/tokenizer.json", "r"))
    start_id = max([e["id"] for e in tok["added_tokens"]]) + 1
    new_tok = {
        "added_tokens": tok["added_tokens"] + add_token(level=4, codebook_size=256, start_id=start_id)
    }
    json.dump(new_tok, open("added_tokens.json", "w"), indent=1)