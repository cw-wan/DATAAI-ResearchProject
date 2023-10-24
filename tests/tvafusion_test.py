from dataloaders.MELD_dataloader import dataloaderMELD
from modules.confede.TVA_fusion import TVAFusion
import torch
from utils.training import eval_tva_fusion

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load train
train_dataloader = dataloaderMELD(datapath="../data/MELD", subset="dev", batch_size=10, shuffle=True)

# load model
model = TVAFusion()
model.config.MELD.Path.checkpoints_path = '../checkpoints/'
model.load_model(load_pretrain=True)
model.to(device)

acc, f1 = eval_tva_fusion(model)
print(acc, f1)
