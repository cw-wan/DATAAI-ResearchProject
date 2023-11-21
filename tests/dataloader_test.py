from dataloaders.MELD_dataloader import dataloaderMELD
from tqdm import tqdm
import os

os.chdir("../")

eval_data = dataloaderMELD(datapath="data/MELD",
                           subset="dev",
                           batch_size=3,
                           shuffle=False)

bar = tqdm(eval_data)

for i, samples in enumerate(bar):
    samples2 = eval_data.dataset.contrastive_sample(samples["index"])
    print(samples["emotion"])
    print(samples2["emotion"])
    break
