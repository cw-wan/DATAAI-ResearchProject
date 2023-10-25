from dataloaders import DatasetMELD

# load dev
dataset = DatasetMELD('../data/MELD', 'train')

print(len(dataset))
print(dataset[len(dataset) - 1]['index'])
