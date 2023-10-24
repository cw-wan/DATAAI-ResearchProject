import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class MELD:
    class Path:
        checkpoints_path = 'checkpoints/'
        raw_data_path = 'data/MELD'

    class Downstream:
        const_heat = 0.5
