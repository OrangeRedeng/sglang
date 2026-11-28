# SPDX-License-Identifier: Apache-2.0
"""Full-window repacking and two-window T2VA prediction fusion."""

from dataclasses import dataclass
import math

import torch

from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function

from .denoise_loop import MiniMaxH3DenoiseBranch
from .packed_sequence import minimax_h3_packed_sequence


@dataclass
class Window:
    video: slice
    audio: torch.Tensor
    branch: MiniMaxH3DenoiseBranch
    timesteps: list
    context: torch.Tensor | None = None


def prepare_windows(
    packed,
    tags,
    embeddings,
    *,
    latent_shape,
    audio_t,
    sigmas,
    overlap,
    device,
    full_window=False,
    offset=0,
    cosine=False,
    anchors=False,
):
    frames, height, width = latent_shape
    frame_rows = packed["img_pos"].numel() // frames
    audio_pos = packed["audio_pos"].reshape(-1, audio_t)
    positions = packed["img_position_ids"]
    if full_window:
        ranges = ((0, frames, 0, audio_t),)
        video_ramp = audio_ramp = None
    else:
        if frames < 4 or not math.isfinite(overlap) or not 0 < overlap <= 0.5:
            raise ValueError(
                "H3 window overlap must be in (0, 0.5], with at least four latent frames"
            )
        overlap_frames = min(frames - 2, max(1, round(frames * overlap)))
        right_start = (frames - overlap_frames) // 2 + offset
        if not 0 < right_start < frames - overlap_frames:
            raise ValueError("H3 window offset leaves an empty exclusive region")
        left_stop = right_start + overlap_frames
        video_times = positions[packed["img_pos"][::frame_rows], 0]
        audio_times = positions[audio_pos[0], 0]
        audio_start = int(torch.searchsorted(audio_times, video_times[right_start]))
        audio_stop = int(torch.searchsorted(audio_times, video_times[left_stop]))
        if not 0 < audio_start < audio_stop < audio_t:
            raise ValueError(
                "H3 window split does not produce overlapping audio ranges"
            )
        ranges = (
            (0, left_stop, 0, audio_stop),
            (right_start, frames, audio_start, audio_t),
        )
        video_ramp = _ramp(overlap_frames, device, cosine).repeat_interleave(frame_rows)
        audio_ramp = _ramp(audio_stop - audio_start, device, cosine)

    windows = []
    for v_start, v_stop, a_start, a_stop in ranges:
        video = slice(v_start * frame_rows, v_stop * frame_rows)
        audio = torch.arange(audio_pos.numel()).reshape_as(audio_pos)[:, a_start:a_stop]
        anchor_frames = (
            [
                i
                for i in (frames // 8, frames // 2, 7 * frames // 8)
                if not v_start <= i < v_stop
            ]
            if anchors
            else []
        )
        context = (
            torch.cat(
                [
                    torch.arange(i * frame_rows, (i + 1) * frame_rows)
                    for i in anchor_frames
                ]
            )
            if anchor_frames
            else None
        )
        layout = minimax_h3_packed_sequence(
            text_len=packed["txt_len"],
            latent_t=v_stop - v_start + len(anchor_frames),
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
                packed["img_pos"][context]
                if context is not None
                else packed["img_pos"][:0],
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
        windows.append(
            Window(
                video,
                audio.to(device),
                branch,
                timesteps,
                context.to(device) if context is not None else None,
            )
        )

    return WindowPredictions(
        tuple(windows),
        video_ramp,
        audio_ramp,
        ranges,
        latent_shape,
        video_times if not full_window else None,
        audio_times if not full_window else None,
    )


def _ramp(length, device, cosine=False):
    t = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length
    return 0.5 - 0.5 * torch.cos(math.pi * t) if cosine else t


def boundary_offsets(frames, overlap):
    shared = min(frames - 2, max(1, round(frames * overlap)))
    left = (frames - shared) // 2
    radius = min(max(1, frames // 20), left - 1, frames - shared - left - 1)
    return (0, radius, -radius) if radius > 0 else (0,)


def boundary_costs(plans, video, audio):
    """Use temporal changes in the last completed scheduler state."""
    frames = next(iter(plans.values())).latent_shape[0]
    v = video.reshape(frames, -1).float()
    channels = next(iter(plans.values())).windows[0].audio.shape[0]
    a = audio.reshape(channels, -1, audio.shape[-1]).float()
    dv = (v[1:] - v[:-1]).square().mean(dim=1)
    da = (a[:, 1:] - a[:, :-1]).square().mean(dim=(0, 2))
    scores = []
    for plan in plans.values():
        (_, stop, _, audio_stop), (start, _, audio_start, _) = plan.ranges
        imbalance = abs(stop - (frames - start)) / frames
        audio_imbalance = abs(audio_stop - (a.shape[1] - audio_start)) / a.shape[1]
        scores.append(
            dv[start - 1 : stop].mean() / dv.mean().clamp_min(1e-12)
            + da[audio_start - 1 : audio_stop].mean() / da.mean().clamp_min(1e-12)
            + 0.5 * (imbalance + audio_imbalance)
            + (audio_stop - audio_start) / a.shape[1]
        )
    return torch.stack(scores)


def choose_boundary(offsets, costs):
    if not all(map(math.isfinite, costs)):
        return 0
    return offsets[min(range(len(costs)), key=costs.__getitem__)]


def _disagreement(left, right):
    left, right = left.float(), right.float()
    return (
        (left - right).norm() / (left.norm() + right.norm()).clamp_min(1e-12)
    ).item()


class WindowPredictions:
    def __init__(
        self,
        windows,
        video_ramp,
        audio_ramp,
        ranges,
        latent_shape,
        video_times,
        audio_times,
    ):
        self.ranges = ranges
        self.latent_shape = latent_shape
        self.video_times = video_times
        self.audio_times = audio_times
        self.windows = windows
        self.video_ramp = video_ramp
        self.audio_ramp = audio_ramp

    def predict(
        self,
        forward,
        model,
        video,
        audio,
        step,
        slot,
        *,
        profile=False,
        diagnostics=None,
    ):
        outputs = []
        for index, window in enumerate(self.windows):
            tag = f"h3_window_s{step}_w{index}"
            with maybe_record_function(f"{tag}_prepare", enabled=profile):
                video_rows = video[window.video]
                if window.context is not None:
                    video_rows = torch.cat(
                        (video_rows, video.index_select(0, window.context))
                    )
                kwargs = window.branch.forward_kwargs(
                    video_rows=video_rows,
                    audio_rows=audio.index_select(0, window.audio.flatten()),
                    step_timesteps=window.timesteps[step],
                    adaln_slot=slot,
                )
            with maybe_record_function(f"{tag}_dit", enabled=profile):
                v, a = forward(model, kwargs, step)
                if window.context is not None:
                    v = v[: window.video.stop - window.video.start]
                outputs.append((v, a))
        if len(outputs) == 1:
            return outputs[0]
        with maybe_record_function(f"h3_window_s{step}_fusion", enabled=profile):
            (left_video, left_audio), (right_video, right_audio) = outputs
            if diagnostics is not None:
                diagnostics.update(
                    video_disagreement=_disagreement(
                        left_video[-self.video_ramp.numel() :],
                        right_video[: self.video_ramp.numel()],
                    ),
                    audio_disagreement=_disagreement(
                        left_audio.reshape(
                            self.windows[0].audio.shape[0], -1, left_audio.shape[-1]
                        )[:, -self.audio_ramp.numel() :],
                        right_audio.reshape(
                            self.windows[0].audio.shape[0], -1, right_audio.shape[-1]
                        )[:, : self.audio_ramp.numel()],
                    ),
                )
            video_prediction = _blend(left_video, right_video, self.video_ramp)
            channels = self.windows[0].audio.shape[0]
            audio_prediction = _blend(
                left_audio.reshape(channels, -1, left_audio.shape[-1]).transpose(0, 1),
                right_audio.reshape(channels, -1, right_audio.shape[-1]).transpose(
                    0, 1
                ),
                self.audio_ramp,
            )
            return video_prediction, audio_prediction.transpose(0, 1).flatten(0, 1)


def _blend(left, right, ramp):
    count = ramp.numel()
    weights = ramp.reshape(count, *([1] * (left.ndim - 1)))
    # Both velocities refer to the same x and sigma, so lerp commutes with x0 = x + sigma*v.
    center = torch.lerp(left[-count:].float(), right[:count].float(), weights)
    return torch.cat((left[:-count].float(), center, right[count:].float()), dim=0)
