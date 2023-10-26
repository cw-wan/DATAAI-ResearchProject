from dataloaders.MELD_dataloader import dataloaderMELD
from modules.confede.TVA_fusion import TVAFusion
import torch
from modules.imagebind.data import load_and_transform_text as tokenize

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load train
dataloader = dataloaderMELD(datapath="../data/MELD", subset="dev", batch_size=1, shuffle=False)

# load model
model = TVAFusion()
model.config.MELD.Path.checkpoints_path = '../checkpoints/'
model.load_model(load_pretrain=True)
model.to(device)
model.eval()

sample = next(iter(dataloader))
print(sample["text"])
text1 = tokenize(sample["text"], device=device)
vision1 = sample["vision"].clone().detach().to(device)
audio1 = sample["audio"].clone().detach().to(device)

with torch.no_grad():
    T, V, A = model.encode(text1, vision1, sample["video_mask"], audio1)
print(T)
print(V)
print(A)
