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