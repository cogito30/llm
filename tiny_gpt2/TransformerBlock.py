import torch
import torch.nn as nn

from LayerNorm import LayerNorm
from FeedForward import FeedForward
from MultiHeadAttention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attentions = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attentions(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


if __name__ == "__main__":
    import cfg

    torch.manual_seed(21)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(cfg.GPT_CONFIG_124M)
    output = block(x)

    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")