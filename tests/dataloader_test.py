from dataloaders import DataloaderMELD

# load dev
dataloader = DataloaderMELD('../data/MELD', 'dev', 1)

print(dataloader[2])
