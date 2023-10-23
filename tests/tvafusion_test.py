from dataloaders import DatasetMELD
from torch.utils.data import DataLoader
from modules.confede.TVA_fusion import TVAFusion
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load dev
meld = DatasetMELD('../data/MELD', 'dev')

dev_dataloader = DataLoader(meld, batch_size=1, shuffle=True)
dev_dataloader2 = DataLoader(meld, batch_size=8, shuffle=True)

sample = next(iter(dev_dataloader))
sample2 = next(iter(dev_dataloader2))

print(sample["emotion"])
print(sample2["emotion"])

# load model
model = TVAFusion()
model.to(device)

loss, pred_loss, c_loss, mono_loss = model(sample, sample2=sample2)
print(loss)
print(pred_loss)
print(c_loss)
print(mono_loss)
