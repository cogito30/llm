import torch

layer = torch.nn.Embedding(6, 3)
print(layer.weight)

out = layer(torch.tensor([3]))
print(out.shape)
print(out)

print(torch.tensor([[0, 3, 2], [1, 2, 3]]).shape)