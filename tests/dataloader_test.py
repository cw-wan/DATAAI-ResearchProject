from dataloaders import DatasetMELD

# load dev
dataset = DatasetMELD('../data/MELD', 'dev')

print(len(dataset))
print(dataset[len(dataset) - 1]['video_mask'])
