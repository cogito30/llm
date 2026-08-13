import torch

def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_specials={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

if __name__ == "__main__":
    from Tokenizer import Tokenizer
    from TinyGPT2 import TinyGPT2
    import cfg

    start_text = "Hello, I am"


    tokenizer = Tokenizer("gpt2")
    encoded = tokenizer.encode(start_text)
    print(encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print(f"encoded shape: {encoded_tensor.shape}")

    model = TinyGPT2(cfg.GPT_CONFIG_124M)
    model.eval()
    out = generate_text(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=cfg.GPT_CONFIG_124M["context_length"])
    print(f"output: {out}")
    print(f"output shape: {out.shape}")
    decoded = tokenizer.decode(out.squeeze(0).tolist())
    print(f"decoded: {decoded}")

    print("---Function Test---")
    start_context = "Every effort moves you"

    token_ids = generate_text(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=256)
    print(f"output: {token_ids_to_text(token_ids, tokenizer)}")


