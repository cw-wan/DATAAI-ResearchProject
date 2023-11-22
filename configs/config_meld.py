import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class MELD:
    class Path:
        imagebind = "modules/imagebind"
        checkpoints_path = 'checkpoints/'
        raw_data_path = 'data/MELD'

    class Downstream:
        const_heat = 0.5

        batch_size = 16
        lr = 1e-4
        epoch = 25
        decay = 1e-3
        num_warm_up = 1
