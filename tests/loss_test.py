from modules.loss import FocalLoss
import torch

# Withoout class weights
criterion = FocalLoss(gamma=0.7)

batch_size = 10
n_class = 5
m = torch.nn.Softmax(dim=-1)
logits = torch.randn(batch_size, n_class)
target = torch.randint(0, n_class, size=(batch_size,))
print(logits)
print(target)
loss = criterion(m(logits), target)

print(loss)
