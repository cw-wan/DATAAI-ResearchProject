from pytorch_metric_learning.losses import NTXentLoss
import torch

loss = NTXentLoss()

e1 = torch.tensor([2, 2, 3])
e2 = torch.tensor([[1, 1, 1], [6, 2, 4], [3, 9, 4]])

embeddings = torch.cat((e1.unsqueeze(0), e2[:]), dim=0)

print(embeddings)

labels = torch.tensor([0, 0, 1, 2])

result = loss(embeddings=embeddings.float(), labels=labels)

print(result)
