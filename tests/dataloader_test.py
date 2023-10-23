from dataloaders import DatasetMELD

# load dev
dataloader = DatasetMELD('../data/MELD', 'dev')

print(dataloader[2])
