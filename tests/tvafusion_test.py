from dataloaders import DatasetMELD
from torch.utils.data import DataLoader
from modules.confede.TVA_fusion import TVAFusion
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load dev
meld = DatasetMELD('../data/MELD', 'dev')

dev_dataloader = DataLoader(meld, batch_size=2, shuffle=True)

sample = next(iter(dev_dataloader))
sample2 = next(iter(dev_dataloader))

print(sample["emotion"])
print(sample2["emotion"])

# load model
model = TVAFusion()
model.to(device)

pred_loss, mono_loss = model(sample, sample2=sample2)
print(pred_loss, mono_loss)
