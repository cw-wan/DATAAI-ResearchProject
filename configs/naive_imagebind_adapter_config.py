import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    imagebind = "modules/imagebind"
    data = "data/MELD"
    save = "checkpoints"


class Model:
    embedding_size = 1024


class DownStream:
    output_size = 7
    dev_train_ratio = 0.1
    batch_size = 32
    learning_rate = 3e-5
    warm_up = 1
    total_epoch = 20
    decay = 1e-3
