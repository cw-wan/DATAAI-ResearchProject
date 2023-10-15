import torch
import torchvision.transforms
from decord import VideoReader, cpu
from torchvision.transforms import Resize, CenterCrop, Normalize
from utils.transforms_video import ToTensorVideo
import torch.nn as nn
import decord
import numpy as np

decord.bridge.set_bridge("torch")


def _load_video(video_path, max_frames, fps):
    # Read video file
    vr = VideoReader(video_path, ctx=cpu(0))
    # Sampling
    v_len = len(vr)
    v_fps = vr.get_avg_fps()
    start, end = 0, v_len
    step = int(v_fps) / fps
    indices = np.arange(start, end, step).astype(int).tolist()
    frames = vr.get_batch(indices)  # (T, H, W, C)

    frames = frames.float().permute(0, 3, 1, 2)  # (T, C, H, W)
    t, _, h, w = frames.size()
    frames_mask = torch.zeros([max_frames, ]).float()
    output = torch.zeros([max_frames, 3, h, w]).float()
    output[0:t] = frames
    frames_mask[0:t] = 1.
    return output, frames_mask


def _transform(imgsize):
    return nn.Sequential(
        Resize(size=[imgsize, ], interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(imgsize),
        ToTensorVideo(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    )


class VideoLoader:
    def __init__(self, imgsize, frames_count, fps):
        self.transform = _transform(imgsize)
        self.frames_count = frames_count
        self.fps = fps

    def load(self, path):
        video_tensor, video_mask = _load_video(path, self.frames_count, self.fps)
        video_tensor = self.transform(video_tensor)
        return video_tensor, video_mask
