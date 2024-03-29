import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    roberta = "modules/roberta-base"
    data = "data/MELD"
    save = "checkpoints"


class Model:
    embedding_size = 768
    loss = "bce"
    gamma = 2


class DownStream:
    output_size = 7
    batch_size = 32
    learning_rate = 1e-5
    warm_up = 1
    total_epoch = 20
    decay = 1e-3
