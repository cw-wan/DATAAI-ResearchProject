from dataloaders import DataloaderMELD

# load dev
dataloader = DataloaderMELD('../data/MELD', 'train')

print(dataloader[2])
