from dataloaders import DatasetMELD

# load dev
dataset = DatasetMELD('../data/MELD', 'dev')

print(dataset[2]["vision"].shape)
print(dataset[2]["video_mask"].shape)
print(dataset[2]["audio"].shape)
print(dataset[2]["emotion"])

sample2 = dataset.sample([2, ])

print(sample2["index"])
print(sample2["emotion"])
print(sample2["vision"].shape)
print(sample2["video_mask"].shape)
print(sample2["audio"].shape)
