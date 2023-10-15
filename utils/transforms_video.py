import torch.nn as nn


class ToTensorVideo(nn.Module):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        super().__init__()

    def __call__(self, clip):
        return clip / 255.0

    def __repr__(self) -> str:
        return self.__class__.__name__
