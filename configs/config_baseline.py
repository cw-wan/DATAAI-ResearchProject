import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class MELD:
    class Path:
        checkpoints_path = 'checkpoints/'
        raw_data_path = 'data/MELD'
        projectors_path = 'checkpoints/projectors_3.pth'

    class Downstream:
        batch_size = 16
        lr = 1e-4
        epoch = 25
        decay = 1e-3
        num_warm_up = 1
