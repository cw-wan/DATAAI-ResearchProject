import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class MELD:
    class Path:
        encoder_path = 'checkpoints/imagebind_huge.pth'
        raw_data_path = 'data/MELD'

    class Downstream:
        const_heat = 0.5
