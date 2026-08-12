import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)    # correction: Bessel's correction 적용 여부
        norm_x = (x- mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    print("---LayerNorm---")
    torch.manual_seed(21)
    batch_example = torch.rand(2, 5)
    print(f"bacth: {batch_example}")

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    print(out_ln)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True)
    print(f"mean: {mean}")
    print(f"var: {var}")
