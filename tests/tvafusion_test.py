from dataloaders import DataloaderMELD
from torch.utils.data import DataLoader
from modules.confede.TVA_fusion import TVAFusion
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load dev
meld = DataloaderMELD('../data/MELD', 'dev')

dev_dataloader = DataLoader(meld, batch_size=2, shuffle=True)

sample = next(iter(dev_dataloader))

print("Input")
print(sample['text'])
print(sample['vision'].shape)
print(sample['video_mask'])
print(sample['audio'].shape)

vision = sample["vision"].clone().detach().to(device)
audio = sample["audio"].clone().detach().to(device)

# load model
model = TVAFusion()
model.to(device)
t_embed, v_embed, a_embed = model.encode(sample['text'], vision, sample['video_mask'], audio)

print("Output")
print(t_embed.shape)
print(v_embed.shape)
print(a_embed.shape)
