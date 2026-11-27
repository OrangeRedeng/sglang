# SPDX-License-Identifier: Apache-2.0
"""Two-window T2VA prediction fusion over a shared H3 latent timeline."""

from dataclasses import dataclass
import math

import torch

from .denoise_loop import MiniMaxH3DenoiseBranch
from .packed_sequence import minimax_h3_packed_sequence


@dataclass
class Window:
    video: slice
    audio: torch.Tensor
    branch: MiniMaxH3DenoiseBranch
    timesteps: list


def prepare_windows(
    packed, tags, embeddings, *, latent_shape, audio_t, sigmas, overlap, device
):
    frames, height, width = latent_shape
    if frames < 4 or not math.isfinite(overlap) or not 0 < overlap <= 0.5:
        raise ValueError(
            "H3 window overlap must be in (0, 0.5], with at least four latent frames"
        )
    overlap_frames = min(frames - 2, max(1, round(frames * overlap)))
    right_start = (frames - overlap_frames) // 2
    left_stop = right_start + overlap_frames
    frame_rows = packed["img_pos"].numel() // frames
    audio_pos = packed["audio_pos"].reshape(-1, audio_t)
    positions = packed["img_position_ids"]
    video_times = positions[packed["img_pos"][::frame_rows], 0]
    audio_times = positions[audio_pos[0], 0]
    audio_start = int(torch.searchsorted(audio_times, video_times[right_start]))
    audio_stop = int(torch.searchsorted(audio_times, video_times[left_stop]))
    if not 0 < audio_start < audio_stop < audio_t:
        raise ValueError("H3 window split does not produce overlapping audio ranges")

    windows = []
    for v_start, v_stop, a_start, a_stop in (
        (0, left_stop, 0, audio_stop),
        (right_start, frames, audio_start, audio_t),
    ):
        video = slice(v_start * frame_rows, v_stop * frame_rows)
        audio = torch.arange(audio_pos.numel()).reshape_as(audio_pos)[:, a_start:a_stop]
        layout = minimax_h3_packed_sequence(
            text_len=packed["txt_len"],
            latent_t=v_stop - v_start,
            latent_h=height,
            latent_w=width,
            audio_t=a_stop - a_start,
            include_keyframe_cond=False,
        )
        # Repack physical rows while keeping the original multimodal RoPE IDs.
        local = torch.cat((layout["text_pos"], layout["audio_pos"], layout["img_pos"]))
        source = torch.cat(
            (
                packed["text_pos"],
                audio_pos[:, a_start:a_stop].flatten(),
                packed["img_pos"][video],
            )
        )
        layout["img_position_ids"][local] = positions[source]
        layout["token_tags"][local] = tags[source]
        branch = MiniMaxH3DenoiseBranch(
            packed=layout,
            text_embeddings=embeddings,
            token_tags=layout["token_tags"],
            device=device,
        )
        timesteps = branch.prepare_timestep_plan(
            video_timesteps=[1.0 - s for s in sigmas["video"][:-1]],
            audio_timesteps=[1.0 - s for s in sigmas["audio"][:-1]],
            imgvid_cond_noise_aug=0.0,
            audio_ref_cond_noise_aug=0.0,
        )
        windows.append(Window(video, audio.to(device), branch, timesteps))

    video_ramp = _ramp(overlap_frames, device).repeat_interleave(frame_rows)
    audio_ramp = _ramp(audio_stop - audio_start, device)
    return WindowPredictions(*windows, video_ramp, audio_ramp)


def _ramp(length, device):
    return (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length


class WindowPredictions:
    def __init__(self, left, right, video_ramp, audio_ramp):
        self.left = left
        self.right = right
        self.video_ramp = video_ramp
        self.audio_ramp = audio_ramp

    @property
    def windows(self):
        return (self.left, self.right)

    def predict(self, forward, model, video, audio, step, slot):
        outputs = []
        for window in self.windows:
            kwargs = window.branch.forward_kwargs(
                video_rows=video[window.video],
                audio_rows=audio.index_select(0, window.audio.flatten()),
                step_timesteps=window.timesteps[step],
                adaln_slot=slot,
            )
            outputs.append(forward(model, kwargs, step))
        (left_video, left_audio), (right_video, right_audio) = outputs
        video_prediction = _blend(left_video, right_video, self.video_ramp)
        channels = self.left.audio.shape[0]
        audio_prediction = _blend(
            left_audio.reshape(channels, -1, left_audio.shape[-1]).transpose(0, 1),
            right_audio.reshape(channels, -1, right_audio.shape[-1]).transpose(0, 1),
            self.audio_ramp,
        )
        return video_prediction, audio_prediction.transpose(0, 1).flatten(0, 1)


def _blend(left, right, ramp):
    count = ramp.numel()
    weights = ramp.reshape(count, *([1] * (left.ndim - 1)))
    # Both velocities refer to the same x and sigma, so lerp commutes with x0 = x + sigma*v.
    center = torch.lerp(left[-count:].float(), right[:count].float(), weights)
    return torch.cat((left[:-count].float(), center, right[count:].float()), dim=0)
