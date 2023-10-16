from dataloaders import DataloaderMELD

dataloader = DataloaderMELD('../data/MELD', 'dev', 1)

print(dataloader[2])
