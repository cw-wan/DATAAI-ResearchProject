# Adapted from https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
import os
import torch
import torchaudio
import pydub
import torchvision.transforms
from decord import VideoReader, cpu
from torchvision.transforms import Resize, CenterCrop, Normalize
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
import torch.nn as nn
import decord
import numpy as np

decord.bridge.set_bridge("torch")

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds


class ToTensorVideo(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, clip):
        return clip / 255.0

    def __repr__(self) -> str:
        return self.__class__.__name__


def _waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    # if abs(p) / n_frames > 0.2:
    #     logging.warning(
    #         "Large gap between audio n_frames(%d) and "
    #         "target_length (%d). Is the audio_target_length "
    #         "setting correct?",
    #         n_frames,
    #         target_length,
    #     )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def _get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def _load_video(video_path,
                max_frames,
                fps,
                num_mel_bins=128,
                target_length=204,
                sample_rate=16000,
                clip_duration=2,
                clips_per_video=3,
                mean=-4.268,
                std=9.138, ):
    # Read video file
    vr = VideoReader(video_path, ctx=cpu(0))
    # Extract audio from video
    audio_path = video_path[:-3] + "wav"
    if not os.path.exists(audio_path):
        video = pydub.AudioSegment.from_file(video_path, format="mp4")
        video.export(audio_path, format="wav")
    waveform, sr = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    # incase the waveform length is too small
    if waveform.shape[1] < clip_duration * sample_rate:
        padding = torch.zeros([waveform.shape[0], clip_duration * sample_rate - waveform.shape[1]])
        waveform = torch.cat([waveform, padding], dim=1)
    os.remove(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    all_clips_timepoints = _get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[  # shape (num_channels, clip_duration * sample_rate)
                        :,
                        int(clip_timepoints[0] * sample_rate): int(
                            clip_timepoints[1] * sample_rate
                        ),
                        ]
        waveform_melspec = _waveform2melspec(  # shape (1, num_mel_bins, target_length)
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)
    normalize = torchvision.transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac) for ac in all_clips]
    audio_output = torch.stack(all_clips, dim=0)
    # Sampling
    v_len = len(vr)
    v_fps = vr.get_avg_fps()
    start, end = 0, v_len
    # Sparse uniform sample strategy
    step = int(v_fps) / fps if max_frames >= fps * v_len / v_fps else int((v_len - 1) / (max_frames - 1))
    indices = np.arange(start, end, step).astype(int).tolist()
    # print(" {}, {}, {}".format(v_len, step, indices))
    frames = vr.get_batch(indices)  # (T, H, W, C)
    frames = frames.float().permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = frames[:max_frames, :]
    t, _, h, w = frames.size()
    frames_mask = torch.zeros([max_frames, ]).float()
    vision_output = torch.zeros([max_frames, 3, h, w]).float()
    vision_output[0:t] = frames
    frames_mask[0:t] = 1
    return vision_output, frames_mask, audio_output


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
        video_tensor, video_mask, audio_tensor = _load_video(path, self.frames_count, self.fps)
        video_tensor = self.transform(video_tensor)
        return video_tensor, video_mask, audio_tensor
