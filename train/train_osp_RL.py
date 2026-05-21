"""
OSPNext RL Post-Training Script (GRPO)

Combines:
  - train_osp.py: FSDP2 distributed infrastructure, OSPNext model, WanVAE, T5Encoder, FlowMatching scheduler
  - train_wan2_1.py: GRPO RL training logic (sampling with log_prob, reward, advantage, PPO clipped loss)
"""

import os
import sys
import copy
import math
import yaml
import time
import json
import random
import tempfile
import contextlib
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent import futures
from functools import partial
from argparse import ArgumentParser

import wandb
import imageio

from ospnext.utils.utils import check_and_import_npu, is_npu_available
import torch
check_and_import_npu()

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from torch.utils.data import DataLoader, Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from ospnext.data import ultra_datasets, ultra_samplers, ultra_collators
from ospnext.data.utils.utils import cyclic_iter
from ospnext.utils.log_utils import get_logger, log_on_main_process, verify_min_gpu_count
from ospnext.utils.random_utils import set_seed
from ospnext.distributed.utils import (
    setup_distributed_env,
    cleanup_distributed_env,
    set_modules_to_forward_prefetch,
    set_modules_to_backward_prefetch,
    gather_data_from_all_ranks,
)
from ospnext.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from ospnext.distributed.fsdp_ema import FSDPEMAModel as EMAModel
from ospnext.distributed.cp_state import cp_state

from ospnext.modules import (
    WanVAE,
    T5EncoderModel,
    models,
    models_main_block,
    models_blocks_to_float,
    models_blocks_to_output_float,
)
from ospnext.schedulers import schedulers

from ospnext.distributed.checkpoint import Checkpointer, PREFIX as checkpoint_prefix
from ospnext.utils.constant import VIDEO, PROMPT, PROMPT_IDS, PROMPT_MASK, START_FRAME
from ospnext.utils.utils import str_to_precision, params_nums_to_str, get_memory_allocated
from ospnext.utils.clip_grads import AdaptiveGradClipper
from ospnext.data.utils.wan_utils import WanTextProcessor
from transformers import AutoTokenizer


# ==================== RL Utilities ====================

def sde_step_with_logprob(
    sigmas_schedule,
    model_output,
    timestep_index,
    sample,
    num_inference_steps,
    prev_sample=None,
    generator=None,
    determistic=False,
    return_dt_and_std_dev_t=False,
):
    """
    Flow matching SDE step with log probability computation.
    Adapted from wan_pipeline_with_logprob.py's sde_step_with_logprob,
    but uses our own sigma schedule instead of diffusers scheduler.

    Args:
        sigmas_schedule: tensor of shape (num_inference_steps + 1,) with sigma values
        model_output: predicted velocity from the model
        timestep_index: int, current step index
        sample: current latent
        num_inference_steps: total number of inference steps
        prev_sample: optional, if provided, compute log_prob against this sample
        generator: optional random generator
        determistic: if True, no noise is added (ODE step)
        return_dt_and_std_dev_t: if True, return additional info
    """
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    sigma = sigmas_schedule[timestep_index]
    sigma_prev = sigmas_schedule[timestep_index + 1]
    sigma_max = sigmas_schedule[0].item()
    sigma_min = sigmas_schedule[-1].item()

    dt = sigma_prev - sigma

    # Reshape for broadcasting: [B, 1, 1, 1, 1] for 5D latents
    sigma_b = sigma.view(1, 1, 1, 1, 1) if sigma.dim() == 0 else sigma.view(-1, 1, 1, 1, 1)
    dt_b = dt.view(1, 1, 1, 1, 1) if dt.dim() == 0 else dt.view(-1, 1, 1, 1, 1)

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma_b
    prev_sample_mean = (
        sample * (1 + std_dev_t ** 2 / (2 * sigma_b) * dt_b)
        + model_output * (1 + std_dev_t ** 2 * (1 - sigma_b) / (2 * sigma_b)) * dt_b
    )

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if prev_sample is None:
        if timestep_index < num_inference_steps - 1:
            variance_noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt_b) * variance_noise
        else:
            # 最后一步（t=0）不加噪，直接使用预测均值，避免残留无法去除的随机噪声
            prev_sample = prev_sample_mean

    if determistic:
        prev_sample = sample + dt_b * model_output

    # Compute log probability
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt_b)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt_b))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1 * dt_b)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1 * dt_b)


@torch.no_grad()
def osp_sample_with_logprob(
    model,
    scheduler,
    vae,
    latent_shape,
    text_embeddings,
    device,
    weight_dtype,
    num_inference_steps=50,
    guidance_scale=5.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    determistic=False,
    kl_reward=0.0,
    ref_model=None,
    cp_group=None,
    sde_steps=None,
):
    """
    Sample from OSPNext model with log probability tracking.

    SDE/ODE hybrid denoising:
        - 前 sde_steps 步（step 0 ~ sde_steps-1）使用 SDE（添加随机噪声），记录 log_prob
        - 剩余步（step sde_steps ~ num_inference_steps-1）使用 ODE（确定性），log_prob 设为 0

    Args:
        sde_steps: int, 前多少步使用 SDE 去噪。默认 None 表示全部使用 SDE。
                   仅 SDE 步的 latent 和 log_prob 会参与后续训练。

    Returns:
        videos: decoded video tensor [B, C, T, H, W] in float, range [-1, 1]
        all_latents: list of latent tensors at each step (仅 SDE 步)
        all_log_probs: list of log_prob tensors at each step (仅 SDE 步)
        all_kl: list of KL divergence tensors at each step (仅 SDE 步)
    """
    if sde_steps is None:
        sde_steps = num_inference_steps
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    # Generate initial noise
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    # CP 组内同步初始噪声，确保序列切分后各卡的数据一致 (from train_osp.py)
    if cp_group is not None:
        torch.distributed.broadcast(latents, src=dist.get_global_rank(cp_group, 0), group=cp_group)

    # Set up sigma schedule
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * 1000.0

    all_latents = [latents]  # 仅记录 SDE 步的 latent（用于训练）
    all_log_probs = []
    all_kl = []

    for i in range(num_inference_steps):
        # 同步确保上一步的 FSDP/CP 异步通信（all_gather, reduce_scatter 等）完全完成，
        # 避免通信 buffer 被 allocator 复用导致 memory corruption（NPU MTE DDR 地址越界）
        torch.cuda.synchronize()

        # 判断当前步是 SDE 还是 ODE
        is_sde_step = (i < sde_steps)

        latents_input = latents.to(weight_dtype)
        t = timesteps[i]
        t_batch = t.expand(B).to(device)


        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = model(
                latents_input,
                t_batch,
                text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        # NPU 上 FSDP all-gather 是异步的，必须在 forward 完成后同步
        # 否则下一次 forward 的 input tensor 可能和未完成的通信 buffer 重叠
        torch.cuda.synchronize()


        if do_cfg and negative_text_embeddings is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_uncond = model(
                    latents_input,
                    t_batch,
                    negative_text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            # 同步 uncond forward 的通信 buffer
            torch.cuda.synchronize()
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            del noise_uncond  # 及时释放 uncond 输出

        latents_ori = latents.clone()

        if is_sde_step:
            # SDE 步：添加随机噪声，记录 log_prob
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                sigmas,
                noise_pred.float(),
                i,
                latents.float(),
                num_inference_steps,
                determistic=False,
            )
        else:
            # ODE 步：确定性去噪，不添加噪声
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                sigmas,
                noise_pred.float(),
                i,
                latents.float(),
                num_inference_steps,
                determistic=True,
            )
        del noise_pred, latents_input  # 及时释放中间 tensor

        # 仅保存 SDE 步的 latent/log_prob/kl，ODE 步不参与训练
        if is_sde_step:
            all_latents.append(latents)
            all_log_probs.append(log_prob)
        elif i == sde_steps:
            # SDE→ODE 的边界：记录 SDE 最后一步产生的 latent 作为 next_latents 的最后一项
            # （已在上一个 SDE 步的 append 中记录）
            pass

        # KL computation against reference model (仅 SDE 步)
        if is_sde_step and kl_reward > 0 and not determistic and ref_model is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                ref_noise_pred = ref_model(
                    latents_ori.to(weight_dtype),
                    t_batch,
                    text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            torch.cuda.synchronize()  # 同步 ref_model forward
            if do_cfg and negative_text_embeddings is not None:
                with torch.autocast("cuda", dtype=weight_dtype):
                    ref_noise_uncond = ref_model(
                        latents_ori.to(weight_dtype),
                        t_batch,
                        negative_text_embeddings,
                        start_frame_latents=start_frame_latents,
                    )
                torch.cuda.synchronize()  # 同步 ref_model uncond forward
                ref_noise_pred = ref_noise_uncond + guidance_scale * (ref_noise_pred - ref_noise_uncond)
                del ref_noise_uncond

            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                sigmas,
                ref_noise_pred.float(),
                i,
                latents_ori.float(),
                num_inference_steps,
                prev_sample=latents.float(),
            )
            del ref_noise_pred
            kl = ((prev_latents_mean - ref_prev_latents_mean) ** 2 / (2 * std_dev_t ** 2))
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
            del ref_prev_latents_mean, ref_std_dev_t
        elif is_sde_step:
            all_kl.append(torch.zeros(B, device=device))
        del latents_ori, prev_latents_mean, std_dev_t

        # 每个采样步结束后，释放显存碎片，防止 NPU 内存池累积导致后续步 MTE 越界
        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # 在 VAE decode 前将 all_latents/log_probs/kl 转到 CPU，释放显存给 VAE decode
    all_latents_cpu = [l.cpu() for l in all_latents]
    all_log_probs_cpu = [lp.cpu() for lp in all_log_probs]
    all_kl_cpu = [k.cpu() for k in all_kl]
    del all_latents, all_log_probs, all_kl
    # 双重同步 + 缓存清理，确保 NPU HCCL buffer 完全释放后再进行 VAE decode
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Decode latents to video
    with torch.no_grad():
        videos = vae.decode(latents)  # [B, C, T, H, W], range [-1, 1]
    del latents  # 释放最后一步的 latent

    return videos, all_latents_cpu, all_log_probs_cpu, all_kl_cpu


@torch.no_grad()
def osp_sample_deterministic(
    model,
    scheduler,
    vae,
    latent_shape,
    text_embeddings,
    device,
    weight_dtype,
    num_inference_steps=50,
    guidance_scale=5.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
):
    """
    Deterministic ODE sampling for evaluation (no log_prob tracking).

    Uses full ODE steps (no SDE noise) for reproducible eval generation.
    Returns only decoded video tensor.
    """
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    # Generate initial noise
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    # Set up sigma schedule
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * 1000.0

    for i in range(num_inference_steps):
        torch.cuda.synchronize()

        latents_input = latents.to(weight_dtype)
        t = timesteps[i]
        t_batch = t.expand(B).to(device)

        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = model(
                latents_input,
                t_batch,
                text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        torch.cuda.synchronize()

        if do_cfg and negative_text_embeddings is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_uncond = model(
                    latents_input,
                    t_batch,
                    negative_text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            torch.cuda.synchronize()
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            del noise_uncond

        # ODE step (deterministic)
        latents, _, _, _ = sde_step_with_logprob(
            sigmas,
            noise_pred.float(),
            i,
            latents.float(),
            num_inference_steps,
            determistic=True,
        )
        del noise_pred, latents_input

        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Decode latents to video
    videos = vae.decode(latents)  # [B, C, T, H, W], range [-1, 1]
    del latents

    return videos


def compute_log_prob_for_training(
    model,
    sample,
    step_idx,
    text_embeddings,
    weight_dtype,
    sigmas_schedule,
    num_inference_steps,
    guidance_scale=1.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    ref_model=None,
):
    """
    Compute log probability for a single denoising step during training.
    """
    do_cfg = guidance_scale > 1.0
    latents_input = sample["latents"][:, step_idx].to(weight_dtype)
    t = (sigmas_schedule[step_idx] * 1000.0).expand(latents_input.shape[0]).to(latents_input.device)

    noise_pred = model(
        latents_input,
        t,
        text_embeddings,
        start_frame_latents=start_frame_latents,
    )

    if do_cfg and negative_text_embeddings is not None:
        # uncond forward 不需要梯度（CFG 只用其做方向引导），
        # 使用 no_grad 避免 gradient checkpoint recompute 时
        # DTensor 权重与普通 Tensor 输入不匹配的问题。
        with torch.no_grad():
            noise_uncond = model(
                latents_input,
                t,
                negative_text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        # CFG 混合：noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        # 先 detach uncond 确保不引入无关梯度图，
        # 然后用 torch.lerp 避免 Python float * DTensor 的兼容性问题
        noise_uncond = noise_uncond.detach()
        noise_pred = torch.lerp(noise_uncond, noise_pred, guidance_scale)

    # 确保 noise_pred 是普通 Tensor（非 DTensor），因为 sde_step_with_logprob
    # 内部的标量运算不兼容 DTensor
    if isinstance(noise_pred, DTensor):
        noise_pred = noise_pred.full_tensor()

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        sigmas_schedule,
        noise_pred.float(),
        step_idx,
        sample["latents"][:, step_idx].float(),
        num_inference_steps,
        prev_sample=sample["next_latents"][:, step_idx].float(),
        return_dt_and_std_dev_t=True,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, dt


class TextPromptDataset(Dataset):
    """Text prompt dataset that tokenizes prompts like t2v_dataset.py."""
    def __init__(self, file_path, text_tokenizer_path, text_max_length=512, return_prompt_mask=True):
        with open(file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=text_max_length,
            return_prompt_mask=return_prompt_mask,
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        prompt_ids, prompt_mask = self.text_processor(prompt)
        return {
            PROMPT: prompt,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": {},
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example[PROMPT] for example in examples]
        prompt_ids = torch.cat([example[PROMPT_IDS] for example in examples], dim=0)
        prompt_mask = torch.cat([example[PROMPT_MASK] for example in examples], dim=0)
        metadatas = [example["metadata"] for example in examples]
        return {
            PROMPT: prompts,
            PROMPT_IDS: prompt_ids,
            PROMPT_MASK: prompt_mask,
            "metadata": metadatas,
        }


class DistributedKRepeatSampler(Sampler):
    """
    Distributed sampler that repeats each sample k times across all ranks.
    Ensures the same prompt appears k times in total for GRPO training.
    """
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, \
            f"k cannot divide n*b, k={k}, num_replicas={num_replicas}, batch_size={batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


class PerPromptStatTracker:
    """Track per-prompt statistics for advantage normalization."""
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        return avg_group_size, len(self.history_prompts)

    def clear(self):
        self.stats = {}


def calculate_zero_std_ratio(prompts, gathered_rewards):
    """Calculate the ratio of prompts with zero reward std."""
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    return zero_std_count / len(prompt_std_devs)


# ==================== Main Training ====================

def main(config):
    logger = get_logger()

    # ========== Config ==========
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "osp_next")
    task = config.get("task", "t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})
    # skiparse 相关 (from train_osp.py)
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_1d = model_config.get("skiparse_1d", False)
    skiparse_2d = model_config.get("skiparse_2d", False)
    num_full_blocks = model_config.get("num_full_blocks", 0)

    # RL config
    rl_config = config.get("rl_config", {})
    num_inference_steps = rl_config.get("num_inference_steps", 20)
    guidance_scale = rl_config.get("guidance_scale", 5.0)
    sample_batch_size = rl_config.get("sample_batch_size", 1)
    train_batch_size = rl_config.get("train_batch_size", 1)
    num_batches_per_epoch = rl_config.get("num_batches_per_epoch", 8)
    num_inner_epochs = rl_config.get("num_inner_epochs", 1)
    num_image_per_prompt = rl_config.get("num_image_per_prompt", 4)
    sample_time_per_prompt = rl_config.get("sample_time_per_prompt", 1)
    timestep_fraction = rl_config.get("timestep_fraction", 1.0)
    clip_range = rl_config.get("clip_range", 1e-4)
    adv_clip_max = rl_config.get("adv_clip_max", 5.0)
    kl_reward = rl_config.get("kl_reward", 0.0)
    kl_beta = rl_config.get("kl_beta", 0.0)
    use_cfg_in_train = rl_config.get("use_cfg_in_train", True)
    per_prompt_stat_tracking = rl_config.get("per_prompt_stat_tracking", True)
    global_std = rl_config.get("global_std", False)
    reward_fn_config = rl_config.get("reward_fn", {})
    prompt_file = rl_config.get("prompt_file", None)
    eval_prompt_file = rl_config.get("eval_prompt_file", None)
    video_height = rl_config.get("height", 480)
    video_width = rl_config.get("width", 832)
    video_num_frames = rl_config.get("num_frames", 81)
    eval_freq = rl_config.get("eval_freq", 25)
    eval_num_steps = rl_config.get("eval_num_steps", 50)
    # SDE/ODE hybrid: 前 sde_steps 步使用 SDE（有噪声），剩余步使用 ODE（确定性）
    # sde_steps=0 表示全部 ODE（纯确定性），sde_steps=num_inference_steps 表示全部 SDE（原始行为）
    sde_steps = rl_config.get("sde_steps", num_inference_steps)  # 默认全 SDE

    # data config (for prompt dataset)
    data_config = config.get("data_config", {})

    # optimizer config
    optimizer_config = config.get("optimizer_config", {})

    # training config
    num_epochs = config.get("num_epochs", 1000)
    gradient_checkpointing = config.get("gradient_checkpointing", False)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    init_max_grad_norm = config.get("init_max_grad_norm", 1.0)
    log_interval = config.get("log_interval", 1)
    save_interval = config.get("save_interval", 10)
    weight_dtype = config.get("weight_dtype", "bfloat16")
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    ema_decay = config.get("ema_decay", 0.9999)
    ema_update_interval = config.get("ema_update_interval", 1)
    use_context_parallel = config.get("use_context_parallel", False)
    use_skiparse_context_parallel = config.get("use_skiparse_context_parallel", False)

    # save config
    output_dir = config.get("output_dir", "./output_rl")

    # ========== Distributed Setup ==========
    setup_distributed_env()
    verify_min_gpu_count()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # wandb
    wandb_config = config.get("wandb_config", {})
    if wandb_config.get("project_name", None) is not None and rank == 0:
        project_name = wandb_config.get("project_name")
        wandb.init(
            project=project_name,
            name=wandb_config.get("exp_name", project_name),
            config=config,
            dir=output_dir,
        )

    # FSDP mesh
    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size:
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning: GPU nums not enough! FSDP size reset to {fsdp_size}!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    # ===== Context Parallelism (CP) init, from train_osp.py =====
    dp_group = dist.group.WORLD  # default world group
    cp_size = 1
    use_context_parallel = use_context_parallel and config.get("cp_size", 1) > 1
    # skiparse cp
    skiparse_cp_size = 1
    use_skiparse_context_parallel = use_skiparse_context_parallel and config.get("skiparse_cp_size", 1) > 1 and sparse_ratio > 1
    use_global_context_parallel = use_context_parallel or use_skiparse_context_parallel
    global_cp_size = 1
    # full cp size
    full_cp_size = 1
    use_full_blocks_context_parallel = use_global_context_parallel and (skiparse_1d or skiparse_2d) and num_full_blocks > 0
    global_cp_group = None

    if use_global_context_parallel:
        if use_context_parallel:
            cp_size = config.get("cp_size", 1)
        if use_skiparse_context_parallel:
            skiparse_cp_size = config.get("skiparse_cp_size", 1)
            if skiparse_1d:
                assert skiparse_cp_size <= sparse_ratio and sparse_ratio % skiparse_cp_size == 0
            elif skiparse_2d:
                assert skiparse_cp_size <= sparse_ratio ** 2 and (sparse_ratio ** 2) % skiparse_cp_size == 0
        global_cp_size = skiparse_cp_size * cp_size
        dp_global_cp_mesh = init_device_mesh("cuda", (world_size // global_cp_size, skiparse_cp_size, cp_size), mesh_dim_names=("dp", "skiparse_cp", "cp"))
        dp_group = dp_global_cp_mesh["dp"].get_group()
        global_cp_group = dp_global_cp_mesh["skiparse_cp", "cp"]._flatten().get_group()
        skiparse_cp_group = dp_global_cp_mesh["skiparse_cp"].get_group()
        full_cp_group = cp_group = dp_global_cp_mesh["cp"].get_group()
        log_on_main_process(logger, f"We use context parallel, global_cp_size: {global_cp_size}, cp_size: {cp_size}, skiparse_cp_size: {skiparse_cp_size}")
        cp_state.reset(global_cp_group=global_cp_group, cp_group=cp_group, skiparse_cp_group=skiparse_cp_group, full_cp_group=full_cp_group)

    log_on_main_process(logger, f"use_global_context_parallel: {use_global_context_parallel}, use_context_parallel: {use_context_parallel}, use_skiparse_context_parallel: {use_skiparse_context_parallel}, use_full_blocks_context_parallel: {use_full_blocks_context_parallel}")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    set_seed(seed, device_specific=False)

    # ========== Init Models ==========
    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device,
    )
    log_on_main_process(logger, f"VAE model initialized, memory: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False),
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder initialized, memory: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing scheduler...")
    scheduler = schedulers[scheduler_config.get("scheduler_name", "flow_matching")](**scheduler_config)

    log_on_main_process(logger, "Initializing diffusion model...")
    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    else:
        log_on_main_process(logger, "Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)

    # CP head count validation and full_blocks_cp_group setup (from train_osp.py)
    if use_context_parallel or use_full_blocks_context_parallel:
        if use_context_parallel and model.num_heads % cp_size != 0:
            raise ValueError(f"When using context parallel, num_heads {model.num_heads} must be multiple of cp_size {cp_size}!")
        if use_full_blocks_context_parallel:
            if global_cp_size <= model.num_heads and model.num_heads % global_cp_size == 0:
                full_cp_size = global_cp_size
            else:
                gcd = math.gcd(model.num_heads, global_cp_size)
                full_cp_size = gcd
            dummy_mesh = init_device_mesh("cuda", (world_size // full_cp_size, full_cp_size), mesh_dim_names=("dummy", "full_cp"))
            full_cp_group = dummy_mesh["full_cp"].get_group()
            cp_state.reset(full_cp_group=full_cp_group)

    model.train()

    # FSDP2 wrap
    FSDP2_mix_wrapper(
        model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        blocks_to_output_float=models_blocks_to_output_float[model_name],
        reshard_after_forward=reshard_after_forward,
        cpu_offload=model_cpu_offload,
    )

    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(seed, device_specific=False)
        model.reset_parameters()

    log_on_main_process(logger, f"Diffusion model initialized, memory: {get_memory_allocated()} GiB")

    if gradient_checkpointing:
        log_on_main_process(logger, "Using gradient checkpointing")
        model.set_gradient_checkpointing(True)

    # EMA
    log_on_main_process(logger, "Initializing EMA model...")
    ema_model = EMAModel(model, decay=ema_decay, update_interval=ema_update_interval)
    log_on_main_process(logger, f"EMA model initialized, memory: {get_memory_allocated()} GiB")

    # Load checkpoint
    save_with_dcp_api = config.get("save_with_dcp_api", False)
    checkpointer = Checkpointer(folder=output_dir, dcp_api=save_with_dcp_api)
    if checkpointer.last_training_iteration is not None:
        log_on_main_process(logger, "Loading model checkpoint...")
        checkpointer.load_model(model)
        log_on_main_process(logger, "Loading EMA model checkpoint...")
        ema_model.store(model)
        checkpointer.load_model(model, ema=True)
        ema_model.model_copy_to_ema(model)
        ema_model.restore(model)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from checkpoint {pretrained_model_dir_or_checkpoint}")
        checkpointer.load_model_from_path(model, pretrained_model_dir_or_checkpoint)
        ema_model.model_copy_to_ema(model)
        has_loaded_pretrained_model = True

    if not has_loaded_pretrained_model:
        log_on_main_process(logger, f"Warning! Training from scratch, pretrained_model_dir_or_checkpoint={pretrained_model_dir_or_checkpoint}")

    # Optimizer
    log_on_main_process(logger, "Initializing optimizer...")
    learning_rate = optimizer_config.get("lr", 1e-5)
    weight_decay_val = optimizer_config.get("weight_decay", 1e-2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_config.get("betas", (0.9, 0.999)),
        weight_decay=weight_decay_val,
        eps=optimizer_config.get("eps", 1e-15),
    )
    adaptive_grad_clipper = AdaptiveGradClipper(
        init_max_grad_norm=init_max_grad_norm,
        model_parallel_group=ddp_fsdp_mesh["fsdp"].get_group(),
    )

    if checkpointer.last_training_iteration is not None:
        checkpointer.load_optim(model, optimizer)
        adaptive_grad_clipper.load(
            output_dir=f"{output_dir}/{checkpoint_prefix}{checkpointer.last_training_iteration:09d}"
        )

    first_epoch = 0 if checkpointer.last_training_iteration is None else checkpointer.last_training_iteration

    set_seed(seed, device_specific=True, process_group=dp_group)  # CP组内seed一致

    # ========== RL Dataset & Reward ====================
    # Import reward function from ospnext
    try:
        import ospnext.rewards.rewards
        reward_fn = getattr(ospnext.rewards.rewards, 'multi_score')(device, reward_fn_config)
    except Exception as e:
        log_on_main_process(logger, f"ERROR: Failed to import ospnext.rewards.rewards: {e}")
        log_on_main_process(logger, "Using dummy reward function — all rewards will be 1.0, training will NOT work!")
        import traceback
        traceback.print_exc()
        def reward_fn(videos, prompts, metadata, only_strict=True):
            B = len(prompts)
            return {"avg": np.ones(B, dtype=np.float32)}, {}

    # Prompt dataset (tokenize prompts in dataset, same as t2v_dataset.py)
    text_tokenizer_path = data_config.get("dataset_config", {}).get("text_tokenizer_path", None)
    text_max_length = data_config.get("dataset_config", {}).get("tokenizer_max_length", text_encoder_config.get("text_len", 512))
    if text_tokenizer_path is None:
        raise ValueError("data_config.dataset_config.text_tokenizer_path must be specified for RL training.")
    if prompt_file is not None:
        train_dataset = TextPromptDataset(
            file_path=prompt_file,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )
    else:
        raise ValueError("prompt_file must be specified for RL training.")

    # Compute dp_size and dp_rank for sampling (CP组内的卡应该看到相同的数据)
    dp_size = dp_group.size() if use_global_context_parallel else world_size
    dp_rank = dist.get_rank(dp_group) if use_global_context_parallel else rank

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=sample_batch_size,
        k=num_image_per_prompt,
        num_replicas=dp_size,
        rank=dp_rank,
        seed=seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=TextPromptDataset.collate_fn,
    )

    # Eval dataset
    test_dataloader = None
    if eval_prompt_file is not None:
        eval_dataset = TextPromptDataset(
            file_path=eval_prompt_file,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )
        test_dataloader = DataLoader(
            eval_dataset,
            batch_size=sample_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=4,
        )

    # Stat tracker
    if num_image_per_prompt * sample_time_per_prompt <= 1:
        per_prompt_stat_tracking = False
    stat_tracker = PerPromptStatTracker(global_std=global_std) if per_prompt_stat_tracking else None

    # Negative text embedding for CFG
    # 使用真实的 NEGATIVE_PROMPT 文本（与推理 pipeline 保持一致），而非全零 token
    log_on_main_process(logger, "Computing negative text embedding...")
    from ospnext.utils.constant import NEGATIVE_PROMPT
    neg_text_processor = WanTextProcessor(
        tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
        model_max_length=text_max_length,
        return_prompt_mask=True,
    )
    neg_prompt_ids, neg_prompt_mask = neg_text_processor(NEGATIVE_PROMPT)
    neg_prompt_ids = neg_prompt_ids.to(device)
    neg_prompt_mask = neg_prompt_mask.to(device)
    with torch.no_grad():
        neg_text_embeddings = text_encoder(neg_prompt_ids, neg_prompt_mask)

    # Number of training timesteps per trajectory
    # 仅对 SDE 步进行训练（前 sde_steps 步），ODE 步不参与训练
    # timestep_fraction 在 SDE 步范围内进一步控制训练比例
    num_train_timesteps = int(sde_steps * timestep_fraction)
    train_timestep_indices = list(range(num_train_timesteps))

    # Set up sigma schedule for sampling
    sigmas_schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas_schedule = shift * sigmas_schedule / (1 + (shift - 1) * sigmas_schedule)

    # Executor for async reward computation
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ========== Logging ==========
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main_process(logger, f"""
    {'=' * 20}Start RL Training (GRPO){'=' * 20}
    Model: {model_name}
    Trainable parameters: {params_nums_to_str(trainable_params)}
    Scheduler: {scheduler_config.get("scheduler_name", "flow_matching")}
    Num epochs: {num_epochs}
    Num inference steps: {num_inference_steps}
    SDE steps: {sde_steps} (step 0~{sde_steps-1}: SDE, step {sde_steps}~{num_inference_steps-1}: ODE)
    Num train timesteps: {num_train_timesteps} (only SDE steps are trained)
    Guidance scale: {guidance_scale}
    Sample batch size per GPU: {sample_batch_size}
    Train batch size per GPU: {train_batch_size}
    Num batches per epoch: {num_batches_per_epoch}
    Num inner epochs: {num_inner_epochs}
    Num image per prompt: {num_image_per_prompt}
    Clip range: {clip_range}
    Adv clip max: {adv_clip_max}
    KL reward: {kl_reward}
    KL beta: {kl_beta}
    Per-prompt stat tracking: {per_prompt_stat_tracking}
    Gradient checkpointing: {gradient_checkpointing}
    Weight dtype: {weight_dtype}
    EMA decay: {ema_decay}
    Learning rate: {learning_rate}
    Gradient accumulation steps: {gradient_accumulation_steps}
    World size: {world_size}
    dp_size: {dp_size}
    cp_size: {cp_size}
    skiparse_cp_size: {skiparse_cp_size}
    global_cp_size: {global_cp_size}
    Use Context Parallel: {use_context_parallel}
    Use Skiparse Context Parallel: {use_skiparse_context_parallel}
    Use Full Blocks Context Parallel: {use_full_blocks_context_parallel}
    Video: {video_num_frames}f x {video_height}h x {video_width}w
    Output dir: {output_dir}
    {'=' * 20}{'=' * len('Start RL Training (GRPO)')}{'=' * 20}
    """)

    # ========== Training Loop ==========
    global_step = first_epoch
    train_iter = iter(train_dataloader)

    # Compute latent shape from video dimensions
    patch_t, patch_h, patch_w = model_config.get("patch_size", (1, 2, 2)) if isinstance(model_config.get("patch_size", (1, 2, 2)), (list, tuple)) else (1, 2, 2)
    vae_temporal_factor = 4  # WanVAE temporal compression
    vae_spatial_factor = 8   # WanVAE spatial compression
    latent_T = (video_num_frames - 1) // vae_temporal_factor + 1
    latent_H = video_height // vae_spatial_factor
    latent_W = video_width // vae_spatial_factor
    latent_C = model_config.get("in_dim", 16)
    latent_shape = (sample_batch_size, latent_C, latent_T, latent_H, latent_W)

    log_on_main_process(logger, f"Latent shape: {latent_shape}")

    for epoch in range(first_epoch, num_epochs):
        # ==================== SAMPLING PHASE ====================
        model.eval()

        # ===== Sample 阶段禁用 CP / skiparse_cp，只用 DDP =====
        # 保存原始 CP 状态，采样结束后恢复
        import ospnext.distributed.cp_state as _cp_state_module
        _saved_cp_state = {
            'global_cp_group': cp_state.global_cp_group,
            'global_cp_rank': cp_state.global_cp_rank,
            'global_cp_size': cp_state.global_cp_size,
            'cp_group': cp_state.cp_group,
            'cp_rank': cp_state.cp_rank,
            'cp_size': cp_state.cp_size,
            'skiparse_cp_group': cp_state.skiparse_cp_group,
            'skiparse_cp_rank': cp_state.skiparse_cp_rank,
            'skiparse_cp_size': cp_state.skiparse_cp_size,
            'full_cp_group': cp_state.full_cp_group,
            'full_cp_rank': cp_state.full_cp_rank,
            'full_cp_size': cp_state.full_cp_size,
            'is_initialized': cp_state.is_initialized,
        }
        _saved_USE_CP = _cp_state_module.USE_CONTEXT_PARALLEL
        _saved_USE_SKIPARSE_CP = _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL
        _saved_USE_FULL_CP = _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL

        if use_global_context_parallel:
            log_on_main_process(logger, "Sample phase: disabling CP/skiparse_cp, using DDP only")
            cp_state.clear()
            # 重置模块级缓存，让 use_context_parallel() 等函数重新判定为 False
            _cp_state_module.USE_CONTEXT_PARALLEL = None
            _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = None
            _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = None
            # CP 状态变更后必须清除 rope/mask/cp 的缓存，
            # 否则 prepare_freqs / mask_preprocessor 会命中旧 CP 状态下的缓存值，
            # 导致 freqs batch 维度与实际 tensor 不匹配 (e.g. B=2 vs freqs_B=4)
            if model.rope_wrapper is not None:
                model.rope_wrapper.cache.clear()
            model.mask_preprocessor.cache.clear()
            model.context_preprocessor.shard_seq_lens_cache.clear()
            # 同步所有 rank，确保 CP 状态切换完成后再开始采样
            # 避免部分 rank 仍在使用旧 CP group 导致通信 buffer 冲突
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # 采样阶段启用 reshard_after_forward=True，避免 FSDP 参数一直保持 unsharded
        # 占用大量显存。因为 eval 模式下没有 backward 触发 reshard，
        # 参数在第一次 forward 后就永远 unsharded，导致后续 batch 的 sampling 和 VAE decode OOM。
        # 注意：必须在 CP 状态切换和 barrier 完成之后才能设置，因为 set_reshard_after_forward
        # 可能触发 FSDP 内部的参数 reshard（涉及 HCCL 通信），如果此时 CP group 还在被使用
        # 会导致通信 buffer 冲突 → NPU MTE DDR 地址越界
        if reshard_after_forward is not None and not reshard_after_forward:
            model.set_reshard_after_forward(True, recurse=True)
        samples = []
        all_prompts = []
        last_videos_cpu = None  # 保存最后一批 videos 到 CPU 用于 wandb 日志
        last_prompts = None    # 保存最后一批 prompts，与 last_videos_cpu 一一对应

        for batch_idx in tqdm(
            range(num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=(rank != 0),
        ):
            train_sampler.set_epoch(epoch * num_batches_per_epoch + batch_idx)
            batch = next(train_iter)
            prompts = batch[PROMPT]
            prompt_ids = batch[PROMPT_IDS].to(device)
            prompt_mask = batch[PROMPT_MASK].to(device)
            prompt_metadata = batch["metadata"]
            all_prompts.extend(prompts)

            # Encode prompts using T5 text encoder
            with torch.no_grad():
                text_embeddings = text_encoder(prompt_ids, prompt_mask)
            # 同步 text_encoder 前向推理的异步操作
            torch.cuda.synchronize()

            # Save/eval checkpoint at beginning of each epoch
            if batch_idx == 0 and epoch % save_interval == 0 and epoch > 0:
                log_on_main_process(logger, f"Saving checkpoint at epoch {epoch}...")
                checkpointer.save(model, optimizer, None, epoch)
                ema_model.store(model)
                ema_model.ema_copy_to_model(model)
                checkpointer.save_ema_model(model, epoch)
                ema_model.restore(model)
                adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{epoch:09d}")
                # checkpoint save 涉及 FSDP all-gather（收集完整参数写入），必须同步
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Skip first 2 epochs (collecting group statistics)
            if epoch < 2:
                continue

            # 在每个 batch 的采样前同步并清理显存，防止 FSDP all-gather/HCCL buffer 累积导致 memory corruption
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Sample multiple times per prompt
            for sample_t in range(sample_time_per_prompt):
                with torch.no_grad():
                    videos, latents_list, log_probs_list, kl_list = osp_sample_with_logprob(
                        model=model,
                        scheduler=scheduler,
                        vae=vae,
                        latent_shape=latent_shape,
                        text_embeddings=text_embeddings,
                        device=device,
                        weight_dtype=weight_dtype,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_text_embeddings=neg_text_embeddings.expand(sample_batch_size, -1, -1),
                        start_frame_latents=None,
                        determistic=False,
                        kl_reward=kl_reward,
                        ref_model=None,  # TODO: pass reference model if kl_reward > 0
                        cp_group=None,  # sample 阶段不用 CP，只用 DDP
                        sde_steps=sde_steps,
                    )

                # Stack latents and log_probs（仅包含 SDE 步）
                # latents_list: sde_steps+1 个元素（初始 + sde_steps 步产出）
                # log_probs_list / kl_list: sde_steps 个元素
                latents_stacked = torch.stack(latents_list, dim=1)  # (B, sde_steps+1, C, T, H, W)
                log_probs_stacked = torch.stack(log_probs_list, dim=1)  # (B, sde_steps)
                kl_stacked = torch.stack(kl_list, dim=1)  # (B, sde_steps)
                # 释放中间变量，避免显存累积
                del latents_list, log_probs_list, kl_list

                # timesteps 仅对应 SDE 步的索引（0 ~ sde_steps-1）
                timesteps_repeated = torch.arange(sde_steps, device=device).unsqueeze(0).expand(
                    sample_batch_size, -1
                )

                # Compute rewards asynchronously
                # videos: [B, C, T, H, W], range [-1, 1] -> [0, 1] for reward
                # 先转到 CPU 再做 float 运算，避免在 GPU 上创建额外的 float32 副本
                videos_cpu = videos.detach().cpu()
                del videos
                videos_for_reward = (videos_cpu.float() + 1.0) / 2.0
                rewards_future = executor.submit(
                    reward_fn, videos_for_reward.numpy(), prompts, prompt_metadata, True
                )
                del videos_for_reward

                # 保存最后一批 videos 和对应 prompts 到 CPU 用于 wandb 日志
                last_videos_cpu = videos_cpu
                last_prompts = list(prompts)

                # latents/log_probs/kl 已经在 CPU 上（osp_sample_with_logprob 内部转移了）
                samples.append({
                    "prompt_embeds": text_embeddings.detach().cpu(),
                    "neg_prompt_embeds": neg_text_embeddings.expand(sample_batch_size, -1, -1).detach().cpu(),
                    "timesteps": timesteps_repeated.cpu(),
                    "latents": latents_stacked[:, :-1].detach(),
                    "next_latents": latents_stacked[:, 1:].detach(),
                    "log_probs": log_probs_stacked.detach(),
                    "kl": kl_stacked.detach(),
                    "rewards": rewards_future,
                })
                # 释放 GPU 上的大 tensor
                del latents_stacked, log_probs_stacked, kl_stacked, videos_cpu
                torch.cuda.empty_cache()

            # 释放本 batch 的 GPU tensor，避免跨 batch 显存残留
            del text_embeddings, prompt_ids, prompt_mask

        # ===== Sample 阶段结束，恢复 CP 状态给训练阶段使用 =====
        if use_global_context_parallel:
            log_on_main_process(logger, "Sample phase done: restoring CP state for training")
            cp_state.global_cp_group = _saved_cp_state['global_cp_group']
            cp_state.global_cp_rank = _saved_cp_state['global_cp_rank']
            cp_state.global_cp_size = _saved_cp_state['global_cp_size']
            cp_state.cp_group = _saved_cp_state['cp_group']
            cp_state.cp_rank = _saved_cp_state['cp_rank']
            cp_state.cp_size = _saved_cp_state['cp_size']
            cp_state.skiparse_cp_group = _saved_cp_state['skiparse_cp_group']
            cp_state.skiparse_cp_rank = _saved_cp_state['skiparse_cp_rank']
            cp_state.skiparse_cp_size = _saved_cp_state['skiparse_cp_size']
            cp_state.full_cp_group = _saved_cp_state['full_cp_group']
            cp_state.full_cp_rank = _saved_cp_state['full_cp_rank']
            cp_state.full_cp_size = _saved_cp_state['full_cp_size']
            cp_state.is_initialized = _saved_cp_state['is_initialized']
            # 恢复模块级缓存
            _cp_state_module.USE_CONTEXT_PARALLEL = _saved_USE_CP
            _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = _saved_USE_SKIPARSE_CP
            _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = _saved_USE_FULL_CP
            # CP 状态变更后必须清除 rope/mask/cp 的缓存，
            # 否则 prepare_freqs / mask_preprocessor 会命中采样阶段（CP 禁用）的缓存值，
            # 导致 freqs batch 维度与实际 tensor 不匹配 (e.g. B=2 vs freqs_B=4)
            if model.rope_wrapper is not None:
                model.rope_wrapper.cache.clear()
            model.mask_preprocessor.cache.clear()
            model.context_preprocessor.shard_seq_lens_cache.clear()
            # 同步所有 rank，确保 CP 状态恢复完成后再进入训练
            dist.barrier()
            torch.cuda.synchronize()

        if epoch < 2:
            continue

        # Wait for all rewards
        for sample in tqdm(samples, desc="Waiting for rewards", disable=(rank != 0)):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        # Collate all samples (keep on CPU, move to GPU per micro-batch during training)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # Log videos periodically
        if epoch % 1 == 0 and rank == 0 and last_videos_cpu is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_vis = min(8, len(last_videos_cpu))
                sample_indices = random.sample(range(len(last_videos_cpu)), num_vis)
                for idx, i in enumerate(sample_indices):
                    video = last_videos_cpu[i].numpy().transpose(1, 2, 3, 0)  # T, H, W, C
                    frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=8, codec="libx264", format='FFMPEG')

                if wandb.run is not None:
                    sampled_prompts = [last_prompts[i] for i in sample_indices]
                    wandb.log(
                        {
                            "videos": [
                                wandb.Video(os.path.join(tmpdir, f"{idx}.mp4"), caption=f"{p:.100}", fps=8)
                                for idx, p in enumerate(sampled_prompts)
                            ],
                        },
                        step=global_step,
                    )

        # Apply KL penalty to rewards
        # Expand avg reward to match timestep dimension: (B,) -> (B, sde_steps)
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        kl_on_device = samples["kl"].to(device)
        num_steps_dim = kl_on_device.shape[1] if kl_on_device.dim() > 1 else sde_steps
        avg_expanded = samples["rewards"]["avg"].unsqueeze(-1).expand(-1, num_steps_dim)
        samples["rewards"]["avg"] = avg_expanded - kl_reward * kl_on_device
        del kl_on_device

        # Gather rewards across all dp ranks (注意：CP组内的卡处理相同数据，只在dp_group间gather)
        gathered_rewards = {}
        for key, value in samples["rewards"].items():
            if value.dim() == 1:
                gathered = gather_data_from_all_ranks(value.unsqueeze(0), dim=0, group=dp_group if use_global_context_parallel else None)
                gathered_rewards[key] = gathered.reshape(-1).cpu().numpy()
            else:
                gathered = gather_data_from_all_ranks(value, dim=0, group=dp_group if use_global_context_parallel else None)
                gathered_rewards[key] = gathered.reshape(-1, *value.shape[1:]).cpu().numpy()

        # Log rewards
        if rank == 0:
            log_dict = {
                "epoch": epoch,
                "kl": samples["kl"].mean().cpu().item(),
                "kl_abs": samples["kl"].abs().mean().cpu().item(),
            }
            for key, value in gathered_rewards.items():
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    log_dict[f"reward_{key}"] = value.mean()
            if wandb.run is not None:
                wandb.log(log_dict, step=global_step)

        # Compute advantages
        if per_prompt_stat_tracking and stat_tracker is not None:
            # Gather original prompt strings from all dp ranks
            gathered_prompts_list = [None] * dp_size
            if use_global_context_parallel:
                dist.all_gather_object(gathered_prompts_list, all_prompts, group=dp_group)
            else:
                dist.all_gather_object(gathered_prompts_list, all_prompts)
            gathered_prompts_decoded = [p for rank_prompts in gathered_prompts_list for p in rank_prompts]
            advantages = stat_tracker.update(gathered_prompts_decoded, gathered_rewards['ori_avg'])

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio = calculate_zero_std_ratio(gathered_prompts_decoded, gathered_rewards)
            if rank == 0 and wandb.run is not None:
                wandb.log({
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                }, step=global_step)
            stat_tracker.clear()
        else:
            avg_rewards = gathered_rewards['ori_avg']
            advantages = (avg_rewards - avg_rewards.mean()) / (avg_rewards.std() + 1e-4)

        # Redistribute advantages to local rank (keep on CPU with other samples)
        # advantages shape: (total_B, num_steps) or (total_B,)
        advantages = torch.as_tensor(advantages).float()
        if advantages.dim() == 1:
            # (total_B,) -> split by dp_size -> (B,) per dp_rank
            local_advantages = advantages.reshape(dp_size, -1)[dp_rank]
            # Expand to (B, num_train_timesteps) so we can index by timestep
            local_advantages = local_advantages.unsqueeze(-1).expand(-1, num_train_timesteps).contiguous()
        else:
            # (total_B, num_steps) -> split by dp_size -> (B, num_steps) per dp_rank
            local_advantages = advantages.reshape(dp_size, -1, *advantages.shape[1:])[dp_rank]
        samples["advantages"] = local_advantages

        if rank == 0:
            log_on_main_process(logger, f"Epoch {epoch} | advantages abs mean: {samples['advantages'].abs().mean().item():.4f} | kl mean: {samples['kl'].mean().item():.4f}")

        del samples["rewards"]

        # Mask out zero-advantage samples
        mask = (samples["advantages"].abs().sum(dim=1) != 0) if samples["advantages"].dim() > 1 else (samples["advantages"].abs() != 0)

        num_batches_total = num_batches_per_epoch * sample_time_per_prompt
        true_count = mask.sum()
        if true_count == 0:
            samples["advantages"] = samples["advantages"] + 1e-6
            mask = torch.ones(len(samples["advantages"]), dtype=torch.bool)

        if true_count % num_batches_total != 0 and true_count > 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches_total - (true_count % num_batches_total)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True

        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size_local = len(samples["timesteps"])
        num_timesteps = samples["timesteps"].shape[1] if samples["timesteps"].dim() > 1 else num_train_timesteps

        # ==================== TRAINING PHASE ====================
        # Following train_wan2_1.py: each timestep is an independent gradient
        # accumulation unit. The optimizer steps every
        # (gradient_accumulation_steps * num_train_timesteps) backward calls,
        # i.e. every gradient_accumulation_steps *samples* worth of timesteps.
        accum_steps_total = gradient_accumulation_steps * num_train_timesteps
        backward_counter = 0  # counts individual loss.backward() calls

        for inner_epoch in range(num_inner_epochs):
            model.train()
            # 训练阶段恢复原始 reshard_after_forward 设置（有 backward 会正常触发 reshard）
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(False, recurse=True)

            # ==================== RECOMPUTE OLD LOG PROBS ====================
            # 在训练环境下（CP 开启、与有梯度计算时相同的计算路径）重新计算旧策略
            # 的 log_probs 作为基准，消除 Sample 阶段（no CP、no grad）与 Train 阶段
            # 之间因 NPU/GPU 算子精度差异导致的固定偏差 δ。
            # 这样 ratio = exp(new_log_prob - old_log_prob) 在第一次训练时 ≈ 1.0，
            # 避免因精度误差导致 clip 被过早触发或 ratio 偏移。
            if rank == 0:
                log_on_main_process(logger, f"Epoch {epoch}.{inner_epoch}: recomputing old log_probs in train environment...")
            recomputed_log_probs = torch.zeros_like(samples["log_probs"])  # (B, num_timesteps), on CPU
            recompute_mb_size = train_batch_size  # 用与训练相同的 micro-batch 大小
            num_recompute_mbs = max(1, total_batch_size_local // recompute_mb_size)

            # 使用 enable_grad 而非 no_grad，保证与训练阶段完全一致的计算路径
            # （包括 gradient_checkpointing 的行为），仅不做 backward。
            # 否则 no_grad + autocast 与 有梯度 + autocast 在 gradient_checkpointing
            # 下走不同代码路径，会导致 ratio 偏离 1.0。
            for rc_mb_idx in tqdm(
                range(num_recompute_mbs),
                desc=f"Epoch {epoch}.{inner_epoch}: recomputing old log_probs",
                disable=(rank != 0),
            ):
                rc_start = rc_mb_idx * recompute_mb_size
                rc_end = min(rc_start + recompute_mb_size, total_batch_size_local)
                rc_batch = {
                    k: (
                        {sub_k: sub_v[rc_start:rc_end].to(device) for sub_k, sub_v in v.items()}
                        if isinstance(v, dict)
                        else v[rc_start:rc_end].to(device)
                    )
                    for k, v in samples.items()
                }

                rc_embeds = rc_batch["prompt_embeds"]
                rc_neg_embeds = rc_batch["neg_prompt_embeds"] if use_cfg_in_train else None

                for j in train_timestep_indices:
                    torch.cuda.synchronize()
                    with torch.autocast("cuda", dtype=weight_dtype):
                        _, rc_log_prob, _, _, _ = compute_log_prob_for_training(
                            model=model,
                            sample=rc_batch,
                            step_idx=j,
                            text_embeddings=rc_embeds,
                            weight_dtype=weight_dtype,
                            sigmas_schedule=sigmas_schedule,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale if use_cfg_in_train else 1.0,
                            negative_text_embeddings=rc_neg_embeds,
                            start_frame_latents=None,
                        )
                    recomputed_log_probs[rc_start:rc_end, j] = rc_log_prob.detach().cpu()

                # 释放计算图和 GPU 显存
                del rc_batch, rc_embeds, rc_neg_embeds
                # 清理 recompute 产生的计算图，避免显存泄漏
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            # 用训练环境下重算的 log_probs 替换 Sample 阶段保存的版本
            samples["log_probs"] = recomputed_log_probs
            if rank == 0:
                log_on_main_process(logger, f"Epoch {epoch}.{inner_epoch}: old log_probs recomputed.")

            # Shuffle samples along batch dimension (samples are on CPU)
            perm = torch.randperm(total_batch_size_local)
            samples = {
                k: (
                    {sub_k: sub_v[perm] for sub_k, sub_v in v.items()}
                    if isinstance(v, dict)
                    else v[perm]
                )
                for k, v in samples.items()
            }

            # 注意：不对时间维度做 shuffle。
            # 因为 compute_log_prob_for_training 使用 sigmas_schedule[step_idx]
            # 作为 timestep（固定映射 j → sigma_j），如果对时间维度 shuffle，
            # latents[:, j] 将不再对应 timestep j，导致 sigma 和 latent 错配，
            # ratio 偏离 1.0。batch 维度的 shuffle 已经提供了足够的随机性。

            # Split into micro-batches
            # Each micro-batch processes train_batch_size samples
            num_micro_batches = max(1, total_batch_size_local // train_batch_size)

            info = defaultdict(list)
            for mb_idx in tqdm(
                range(num_micro_batches),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                disable=(rank != 0),
            ):
                mb_start = mb_idx * train_batch_size
                mb_end = min(mb_start + train_batch_size, total_batch_size_local)
                micro_batch = {
                    k: (
                        {sub_k: sub_v[mb_start:mb_end].to(device) for sub_k, sub_v in v.items()}
                        if isinstance(v, dict)
                        else v[mb_start:mb_end].to(device)
                    )
                    for k, v in samples.items()
                }

                embeds = micro_batch["prompt_embeds"]
                neg_embeds = micro_batch["neg_prompt_embeds"] if use_cfg_in_train else None

                for j in train_timestep_indices:
                    # NPU: 确保上一个 timestep 的 backward 通信完成
                    torch.cuda.synchronize()
                    with torch.autocast("cuda", dtype=weight_dtype):
                        prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = compute_log_prob_for_training(
                            model=model,
                            sample=micro_batch,
                            step_idx=j,
                            text_embeddings=embeds,
                            weight_dtype=weight_dtype,
                            sigmas_schedule=sigmas_schedule,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale if use_cfg_in_train else 1.0,
                            negative_text_embeddings=neg_embeds,
                            start_frame_latents=None,
                        )

                    # GRPO loss
                    if micro_batch["advantages"].dim() > 1:
                        adv = torch.clamp(micro_batch["advantages"][:, j], -adv_clip_max, adv_clip_max)
                    else:
                        adv = torch.clamp(micro_batch["advantages"], -adv_clip_max, adv_clip_max)

                    ratio = torch.exp(log_prob - micro_batch["log_probs"][:, j])

                    if epoch == 2 and inner_epoch == 0 and global_step == 0: # 你的代码前两轮跳过了
                        print(f"DEBUG: train_log_prob: {log_prob.mean().item()}, sample_log_prob: {micro_batch['log_probs'][:, j].mean().item()}, ratio: {ratio.mean().item()}")

                    unclipped_loss = -adv * ratio
                    clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    # Optional KL regularization during training
                    if kl_beta > 0:
                        # Use pre-computed KL from sampling
                        kl_loss = micro_batch["kl"][:, j].mean() if micro_batch["kl"].dim() > 1 else micro_batch["kl"].mean()
                        loss = policy_loss + kl_beta * kl_loss
                        info["kl_loss"].append(kl_loss.detach())
                    else:
                        loss = policy_loss

                    # Scale loss for gradient accumulation across timesteps AND samples
                    loss = loss / accum_steps_total
                    loss.backward()

                    info["approx_kl"].append(
                        0.5 * torch.mean((log_prob - micro_batch["log_probs"][:, j]) ** 2).detach()
                    )
                    info["clipfrac"].append(
                        torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).detach()
                    )
                    info["policy_loss"].append(policy_loss.detach())
                    info["loss"].append(loss.detach())

                    backward_counter += 1

                    # Optimizer step after accumulating enough gradients
                    # This fires every accum_steps_total backward calls,
                    # i.e. every gradient_accumulation_steps samples × num_train_timesteps timesteps
                    if backward_counter % accum_steps_total == 0:
                        grad_norm = adaptive_grad_clipper.adaptive_clip(model.parameters())
                        optimizer.step()
                        optimizer.zero_grad()
                        ema_model.update(model, global_step + 1)
                        global_step += 1

                        # Log
                        if len(info) > 0:
                            info_mean = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                            if rank == 0:
                                tqdm.write(
                                    f"  step {global_step} | loss: {info_mean.get('loss', 0):.6f} | "
                                    f"policy_loss: {info_mean.get('policy_loss', 0):.6f} | "
                                    f"approx_kl: {info_mean.get('approx_kl', 0):.6f} | "
                                    f"clipfrac: {info_mean.get('clipfrac', 0):.4f} | "
                                    f"grad_norm: {grad_norm.item():.4f}"
                                )
                                if wandb.run is not None:
                                    wandb_log = {
                                        "train/loss": info_mean.get("loss", 0),
                                        "train/policy_loss": info_mean.get("policy_loss", 0),
                                        "train/approx_kl": info_mean.get("approx_kl", 0),
                                        "train/clipfrac": info_mean.get("clipfrac", 0),
                                        "train/grad_norm": grad_norm.item(),
                                        "train/lr": optimizer.param_groups[0]['lr'],
                                    }
                                    if "kl_loss" in info_mean:
                                        wandb_log["train/kl_loss"] = info_mean["kl_loss"]
                                    wandb_log.update(adaptive_grad_clipper.state_dict())
                                    wandb.log(wandb_log, step=global_step)
                            info = defaultdict(list)

            # Handle remaining gradients at end of inner epoch if not yet stepped
            if backward_counter % accum_steps_total != 0:
                grad_norm = adaptive_grad_clipper.adaptive_clip(model.parameters())
                optimizer.step()
                optimizer.zero_grad()
                ema_model.update(model, global_step + 1)
                global_step += 1
                backward_counter = 0

                if len(info) > 0:
                    info_mean = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                    if rank == 0:
                        tqdm.write(
                            f"  step {global_step} (tail) | loss: {info_mean.get('loss', 0):.6f} | "
                            f"policy_loss: {info_mean.get('policy_loss', 0):.6f} | "
                            f"approx_kl: {info_mean.get('approx_kl', 0):.6f} | "
                            f"clipfrac: {info_mean.get('clipfrac', 0):.4f} | "
                            f"grad_norm: {grad_norm.item():.4f}"
                        )
                        if wandb.run is not None:
                            wandb_log = {
                                "train/loss": info_mean.get("loss", 0),
                                "train/policy_loss": info_mean.get("policy_loss", 0),
                                "train/approx_kl": info_mean.get("approx_kl", 0),
                                "train/clipfrac": info_mean.get("clipfrac", 0),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": optimizer.param_groups[0]['lr'],
                            }
                            if "kl_loss" in info_mean:
                                wandb_log["train/kl_loss"] = info_mean["kl_loss"]
                            wandb_log.update(adaptive_grad_clipper.state_dict())
                            wandb.log(wandb_log, step=global_step)
                    info = defaultdict(list)

        # 训练阶段结束后，同步所有 NPU 异步操作（backward reduce-scatter、optimizer 通信等）
        # 并清理显存缓存，防止残留的 HCCL buffer 在下一轮采样阶段的 model.eval() / 
        # set_reshard_after_forward(True) 时触发 MTE DDR 地址越界
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()

        # Save checkpoint periodically
        if epoch > 0 and epoch % save_interval == 0:
            log_on_main_process(logger, f"Saving checkpoint at epoch {epoch} (global_step {global_step})...")
            checkpointer.save(model, optimizer, None, global_step)
            ema_model.store(model)
            ema_model.ema_copy_to_model(model)
            checkpointer.save_ema_model(model, global_step)
            ema_model.restore(model)
            adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{global_step:09d}")
            # checkpoint 保存涉及 FSDP all-gather（收集完整参数写入），必须同步
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # ==================== EVAL PHASE ====================
        # 在 eval_freq 整除 global_step 时，使用 EMA 模型对 test.txt 中的 prompt 生成视频并 log 到 wandb
        if (test_dataloader is not None
                and global_step > 0
                and global_step % eval_freq == 0):
            log_on_main_process(logger, f"[Eval] Running eval video generation at global_step {global_step}...")
            model.eval()

            # 切换到 EMA 权重
            ema_model.store(model)
            ema_model.ema_copy_to_model(model)

            # Eval 阶段同样需要 reshard_after_forward=True 防止显存 OOM
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(True, recurse=True)

            # 禁用 CP（与采样阶段一致）
            if use_global_context_parallel:
                cp_state.clear()
                _cp_state_module.USE_CONTEXT_PARALLEL = None
                _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = None
                _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = None
                if model.rope_wrapper is not None:
                    model.rope_wrapper.cache.clear()
                model.mask_preprocessor.cache.clear()
                model.context_preprocessor.shard_seq_lens_cache.clear()
                dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            eval_videos_cpu = []
            eval_prompts = []

            for eval_batch in tqdm(test_dataloader, desc=f"[Eval] Generating videos", disable=(rank != 0)):
                eval_prompt_texts = eval_batch[PROMPT]
                eval_prompt_ids = eval_batch[PROMPT_IDS].to(device)
                eval_prompt_mask = eval_batch[PROMPT_MASK].to(device)

                with torch.no_grad():
                    eval_text_embeddings = text_encoder(eval_prompt_ids, eval_prompt_mask)
                torch.cuda.synchronize()

                eval_latent_shape = (len(eval_prompt_texts), latent_C, latent_T, latent_H, latent_W)

                with torch.no_grad():
                    eval_videos = osp_sample_deterministic(
                        model=model,
                        scheduler=scheduler,
                        vae=vae,
                        latent_shape=eval_latent_shape,
                        text_embeddings=eval_text_embeddings,
                        device=device,
                        weight_dtype=weight_dtype,
                        num_inference_steps=eval_num_steps,
                        guidance_scale=guidance_scale,
                        negative_text_embeddings=neg_text_embeddings.expand(len(eval_prompt_texts), -1, -1),
                        start_frame_latents=None,
                    )

                eval_videos_cpu.append(eval_videos.detach().cpu())
                eval_prompts.extend(eval_prompt_texts)

                del eval_videos, eval_text_embeddings, eval_prompt_ids, eval_prompt_mask
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # 恢复 CP 状态
            if use_global_context_parallel:
                cp_state.global_cp_group = _saved_cp_state['global_cp_group']
                cp_state.global_cp_rank = _saved_cp_state['global_cp_rank']
                cp_state.global_cp_size = _saved_cp_state['global_cp_size']
                cp_state.cp_group = _saved_cp_state['cp_group']
                cp_state.cp_rank = _saved_cp_state['cp_rank']
                cp_state.cp_size = _saved_cp_state['cp_size']
                cp_state.skiparse_cp_group = _saved_cp_state['skiparse_cp_group']
                cp_state.skiparse_cp_rank = _saved_cp_state['skiparse_cp_rank']
                cp_state.skiparse_cp_size = _saved_cp_state['skiparse_cp_size']
                cp_state.full_cp_group = _saved_cp_state['full_cp_group']
                cp_state.full_cp_rank = _saved_cp_state['full_cp_rank']
                cp_state.full_cp_size = _saved_cp_state['full_cp_size']
                cp_state.is_initialized = _saved_cp_state['is_initialized']
                _cp_state_module.USE_CONTEXT_PARALLEL = _saved_USE_CP
                _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = _saved_USE_SKIPARSE_CP
                _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = _saved_USE_FULL_CP
                if model.rope_wrapper is not None:
                    model.rope_wrapper.cache.clear()
                model.mask_preprocessor.cache.clear()
                model.context_preprocessor.shard_seq_lens_cache.clear()
                dist.barrier()
                torch.cuda.synchronize()

            # 恢复 reshard_after_forward 设置
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(reshard_after_forward, recurse=True)

            # 恢复训练权重
            ema_model.restore(model)

            # Log eval videos to wandb (only on rank 0)
            if rank == 0 and len(eval_videos_cpu) > 0:
                eval_all_videos = torch.cat(eval_videos_cpu, dim=0)
                with tempfile.TemporaryDirectory() as tmpdir:
                    num_eval_vis = len(eval_all_videos)
                    for idx in range(num_eval_vis):
                        video = eval_all_videos[idx].numpy().transpose(1, 2, 3, 0)  # T, H, W, C
                        frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                        imageio.mimsave(
                            os.path.join(tmpdir, f"eval_{idx}.mp4"),
                            frames, fps=8, codec="libx264", format='FFMPEG',
                        )

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "eval/videos": [
                                    wandb.Video(
                                        os.path.join(tmpdir, f"eval_{idx}.mp4"),
                                        caption=f"{eval_prompts[idx]:.100}",
                                        fps=8,
                                    )
                                    for idx in range(num_eval_vis)
                                ],
                            },
                            step=global_step,
                        )
                del eval_all_videos
            del eval_videos_cpu, eval_prompts
            torch.cuda.empty_cache()

            log_on_main_process(logger, f"[Eval] Eval video generation done at global_step {global_step}.")

    # ========== Final save ==========
    log_on_main_process(logger, f"Saving final checkpoint at global_step {global_step}...")
    checkpointer.save(model, optimizer, None, global_step)
    ema_model.store(model)
    ema_model.ema_copy_to_model(model)
    checkpointer.save_ema_model(model, global_step)
    ema_model.restore(model)
    adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{global_step:09d}")

    log_on_main_process(logger, f"""
    {'=' * 20}End RL Training{'=' * 20}
    Total epochs: {epoch + 1}
    Total global steps: {global_step}
    Model saved to {output_dir}
    {'=' * 20}{'=' * len('End RL Training')}{'=' * 20}
    """)
    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/t2v_rl.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} does not exist!")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
