import torch

model = UNet()

inp = torch.rand([1, 3, 512, 512], dtype=torch.float32)
pred = model(inp)
print(pred.shape)

