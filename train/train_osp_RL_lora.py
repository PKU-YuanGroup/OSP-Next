"""
OSPNext RL Post-Training Script (GRPO) with LoRA + FSDP2

Based on train_osp_RL.py's FSDP2 distributed infrastructure, but uses LoRA
for parameter-efficient training.

Key design:
  - FSDP2 for distributed training (same as train_osp_RL.py / train.py)
  - peft LoRA for parameter-efficient fine-tuning
  - ref_model via disable_adapter() instead of separate model copy
  - FSDPEMAModel for EMA (same as train_osp_RL.py)
  - Checkpointer for checkpoint management (same as train_osp_RL.py)
  - AdaptiveGradClipper for gradient clipping (same as train_osp_RL.py)
"""

import os
import sys
import math
import yaml
import time
import json
import random
import tempfile
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent import futures
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
from torch.utils.data.distributed import DistributedSampler

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
from ospnext.utils.constant import PROMPT, PROMPT_IDS, PROMPT_MASK
from ospnext.utils.utils import str_to_precision, params_nums_to_str, get_memory_allocated
from ospnext.utils.clip_grads import AdaptiveGradClipper
from ospnext.data.utils.wan_utils import WanTextProcessor
from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model, PeftModel


def get_ddp_rank_and_fsdp_local_rank(rank, fsdp_size, world_size):
    """
    Assuming mesh shape is (ddp_size, fsdp_size) and global ranks are laid out contiguously:
        global_rank = ddp_rank * fsdp_size + fsdp_local_rank

    Returns:
        ddp_rank: index of data-parallel replica
        fsdp_local_rank: local rank inside one FSDP replica
    """
    ddp_size = max(1, world_size // fsdp_size)
    ddp_rank = rank // fsdp_size
    fsdp_local_rank = rank % fsdp_size
    return ddp_rank, fsdp_local_rank, ddp_size

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
    cp_group=None,
):
    """
    Flow matching SDE step with log probability computation.
    Adapted from train_osp_RL.py.
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
            # 同步SDE噪声到CP组内
            if cp_group is not None:
                torch.distributed.broadcast(variance_noise, src=dist.get_global_rank(cp_group, 0), group=cp_group)
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
    cp_group=None,
    sde_steps=None,
):
    """
    Sample from OSPNext model with log probability tracking.
    For LoRA: uses disable_adapter() for ref model KL computation.

    SDE/ODE hybrid denoising:
        - 前 sde_steps 步（step 0 ~ sde_steps-1）使用 SDE（添加随机噪声），记录 log_prob
        - 剩余步（step sde_steps ~ num_inference_steps-1）使用 ODE（确定性），log_prob 设为 0

    Args:
        sde_steps: int, 前多少步使用 SDE 去噪。默认 None 表示全部使用 SDE。
                   仅 SDE 步的 latent 和 log_prob 会参与后续训练。

    Returns:
        videos: decoded video tensor [B, C, T, H, W] in float, range [-1, 1]
        all_latents: list of latent tensors at each step (仅 SDE 步, on CPU)
        all_log_probs: list of log_prob tensors at each step (仅 SDE 步, on CPU)
        all_kl: list of KL divergence tensors at each step (仅 SDE 步, on CPU)
    """
    if sde_steps is None:
        sde_steps = num_inference_steps
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    # Generate initial noise
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    # CP 组内同步初始噪声
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
                cp_group=cp_group,
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
                cp_group=cp_group,
            )
        del noise_pred, latents_input

        # 仅保存 SDE 步的 latent/log_prob/kl，ODE 步不参与训练
        if is_sde_step:
            all_latents.append(latents)
            all_log_probs.append(log_prob)

        # KL computation against reference model (LoRA: disable adapter) — 仅 SDE 步
        if is_sde_step and kl_reward > 0 and not determistic:
            with model.disable_adapter():
                with torch.autocast("cuda", dtype=weight_dtype):
                    ref_noise_pred = model(
                        latents_ori.to(weight_dtype),
                        t_batch,
                        text_embeddings,
                        start_frame_latents=start_frame_latents,
                    )
                torch.cuda.synchronize()
                if do_cfg and negative_text_embeddings is not None:
                    with torch.autocast("cuda", dtype=weight_dtype):
                        ref_noise_uncond = model(
                            latents_ori.to(weight_dtype),
                            t_batch,
                            negative_text_embeddings,
                            start_frame_latents=start_frame_latents,
                        )
                    torch.cuda.synchronize()
                    ref_noise_pred = ref_noise_uncond + guidance_scale * (ref_noise_pred - ref_noise_uncond)
                    del ref_noise_uncond

            _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                sigmas,
                ref_noise_pred.float(),
                i,
                latents_ori.float(),
                num_inference_steps,
                prev_sample=latents.float(),
                cp_group=cp_group,
            )
            del ref_noise_pred
            kl = ((prev_latents_mean - ref_prev_latents_mean) ** 2 / (2 * std_dev_t ** 2))
            kl = kl.mean(dim=tuple(range(1, kl.ndim)))
            all_kl.append(kl)
            del ref_prev_latents_mean, ref_std_dev_t
        elif is_sde_step:
            all_kl.append(torch.zeros(B, device=device))
        del latents_ori, prev_latents_mean, std_dev_t

        # 定期清理显存
        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # 在 VAE decode 前将 all_latents/log_probs/kl 转到 CPU，释放显存给 VAE decode
    all_latents_cpu = [l.cpu() for l in all_latents]
    all_log_probs_cpu = [lp.cpu() for lp in all_log_probs]
    all_kl_cpu = [k.cpu() for k in all_kl]
    del all_latents, all_log_probs, all_kl
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Decode latents to video
    with torch.no_grad():
        videos = vae.decode(latents)  # [B, C, T, H, W], range [-1, 1]
    del latents

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
            cp_group=None,
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
    
    # --- DEBUG: 验证 model forward 输出的类型和梯度状态 ---
    if torch.distributed.get_rank() == 0 and step_idx == 0:
        print(f"[DEBUG fwd] noise_pred type={type(noise_pred).__name__}, "
              f"requires_grad={noise_pred.requires_grad}, "
              f"is_DTensor={isinstance(noise_pred, DTensor)}")
    # --- END DEBUG ---

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
        # 先 detach uncond 确保不引入无关梯度图，
        # 然后用 torch.lerp 避免 Python float * DTensor 的兼容性问题
        noise_uncond = noise_uncond.detach()
        noise_pred = torch.lerp(noise_uncond, noise_pred, guidance_scale)

    # 确保 noise_pred 是普通 Tensor（非 DTensor），因为 sde_step_with_logprob
    # 内部的标量运算不兼容 DTensor。
    # 注意: DTensor.full_tensor() 在某些环境（如 NPU）下可能不支持 autograd backward，
    # 导致梯度无法回传到 LoRA 参数。使用 redistribute + _local_tensor 保留梯度图。
    # if isinstance(noise_pred, DTensor):
    #     from torch.distributed.tensor.placement_types import Replicate
    #     noise_pred = noise_pred.redistribute(placements=[Replicate()])._local_tensor

    if isinstance(noise_pred, DTensor):
        noise_pred = noise_pred.full_tensor()  # 保留完整的 autograd 链路

    prev_sample, log_prob, prev_sample_mean, std_dev_t, dt = sde_step_with_logprob(
        sigmas_schedule,
        noise_pred.float(),
        step_idx,
        sample["latents"][:, step_idx].float(),
        num_inference_steps,
        prev_sample=sample["next_latents"][:, step_idx].float(),
        return_dt_and_std_dev_t=True,
        cp_group=None,
    )

    # --- DEBUG: 验证 log_prob 梯度链路 ---
    if torch.distributed.get_rank() == 0 and step_idx == 0:
        _diff = (sample["next_latents"][:, step_idx].float().detach() - prev_sample_mean.detach())
        print(f"[DEBUG chain] noise_pred.requires_grad={noise_pred.requires_grad}, "
              f"log_prob.requires_grad={log_prob.requires_grad}, "
              f"log_prob.grad_fn={log_prob.grad_fn}, "
              f"|prev_sample - prev_sample_mean|={_diff.abs().mean().item():.8f}")
    # --- END DEBUG ---

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


def save_lora_checkpoint(model, save_dir, global_step):
    """Save LoRA weights only (for LoRA-specific checkpoint, rank 0 only).
    
    NOTE: We manually extract and clone LoRA parameters instead of using
    model.save_pretrained(), because after FSDP's _get_full_model_state_dict()
    the underlying tensor storage may become invalid, causing
    'RuntimeError: Attempted to access the data pointer on an invalid python storage.'
    """
    from torch.distributed.tensor import DTensor
    
    save_root = os.path.join(save_dir, f"lora-checkpoint-{global_step}")
    if dist.get_rank() == 0:
        os.makedirs(save_root, exist_ok=True)
    dist.barrier()
    
    # Manually extract LoRA parameters (keys containing 'lora_')
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            # Handle FSDP DTensor: full_tensor() requires all ranks to participate
            if isinstance(param, DTensor):
                full_param = param.full_tensor()
            else:
                full_param = param
                
            if dist.get_rank() == 0:
                # Clone and move to CPU to avoid invalid storage issues
                lora_state_dict[name] = full_param.detach().clone().cpu()
    
    if dist.get_rank() == 0 and lora_state_dict:
        torch.save(lora_state_dict, os.path.join(save_root, "adapter_model.bin"))
        
        # --- DEBUG: print LoRA param norms to verify they changed ---
        for _name, _val in list(lora_state_dict.items())[:3]:
            print(f"[DEBUG save_lora] {_name}: norm={_val.norm().item():.8f}, mean={_val.mean().item():.10f}")
        # --- END DEBUG ---
        
        # Also save the adapter config if available
        if hasattr(model, 'peft_config'):
            import json
            for adapter_name, peft_cfg in model.peft_config.items():
                config_dict = peft_cfg.to_dict() if hasattr(peft_cfg, 'to_dict') else vars(peft_cfg)
                with open(os.path.join(save_root, "adapter_config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)
                break  # Save config for the first (default) adapter
        
        print(f"[Rank 0] LoRA checkpoint saved to {save_root} ({len(lora_state_dict)} parameters)")


RL_TRAINING_STATE_FILE = "rl_training_state.json"


def save_rl_training_state(save_dir, global_step, next_epoch):
    """Persist RL resume metadata alongside each checkpoint."""
    save_root = os.path.join(save_dir, f"{checkpoint_prefix}{global_step:09d}")
    if dist.get_rank() == 0:
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, RL_TRAINING_STATE_FILE), "w", encoding="ascii") as f:
            json.dump(
                {
                    "global_step": int(global_step),
                    "next_epoch": int(next_epoch),
                },
                f,
                indent=2,
            )
    dist.barrier()


def load_rl_training_state(save_dir, global_step):
    """Load RL resume metadata for epoch/global_step bookkeeping."""
    state_path = os.path.join(
        save_dir,
        f"{checkpoint_prefix}{global_step:09d}",
        RL_TRAINING_STATE_FILE,
    )
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="ascii") as f:
        return json.load(f)



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
    # skiparse 相关
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_1d = model_config.get("skiparse_1d", False)
    skiparse_2d = model_config.get("skiparse_2d", False)
    num_full_blocks = model_config.get("num_full_blocks", 0)

    # LoRA config
    lora_config = config.get("lora_config", {})
    lora_rank = lora_config.get("rank", 32)
    lora_alpha = lora_config.get("alpha", 64)
    lora_target_modules = lora_config.get("target_modules", [
        "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
        "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    ])
    lora_path = lora_config.get("lora_path", None)

    # RL config
    rl_config = config.get("rl_config", {})
    num_inference_steps = rl_config.get("num_inference_steps", 20)
    guidance_scale = rl_config.get("guidance_scale", 5.0)
    sample_batch_size = rl_config.get("sample_batch_size", 4)
    train_batch_size = rl_config.get("train_batch_size", 4)
    num_batches_per_epoch = rl_config.get("num_batches_per_epoch", 4)
    num_inner_epochs = rl_config.get("num_inner_epochs", 1)
    num_image_per_prompt = rl_config.get("num_image_per_prompt", 4)
    sample_time_per_prompt = rl_config.get("sample_time_per_prompt", 1)
    timestep_fraction = rl_config.get("timestep_fraction", 1.0)
    clip_range = rl_config.get("clip_range", 5e-3)
    adv_clip_max = rl_config.get("adv_clip_max", 5.0)
    kl_reward = rl_config.get("kl_reward", 0.0)
    kl_beta = rl_config.get("kl_beta", 0.0)
    use_cfg_in_train = rl_config.get("use_cfg_in_train", True)
    per_prompt_stat_tracking = rl_config.get("per_prompt_stat_tracking", True)
    global_std = rl_config.get("global_std", False)
    reward_fn_config = rl_config.get("reward_fn", {})
    prompt_file = rl_config.get("prompt_file", None)
    eval_prompt_file = rl_config.get("eval_prompt_file", None)
    video_height = rl_config.get("height", 720)
    video_width = rl_config.get("width", 1280)
    video_num_frames = rl_config.get("num_frames", 81)
    eval_freq = rl_config.get("eval_freq", 10000)
    eval_num_steps = rl_config.get("eval_num_steps", 50)
    # SDE/ODE hybrid: 前 sde_steps 步使用 SDE（有噪声），剩余步使用 ODE（确定性）
    sde_steps = rl_config.get("sde_steps", num_inference_steps)  # 默认全 SDE

    # EMA config
    ema_decay = config.get("ema_decay", 0.9999)
    ema_update_interval = config.get("ema_update_interval", 1)

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
    save_interval = config.get("save_interval", 100)
    weight_dtype = config.get("weight_dtype", "bfloat16")
    resume_epoch = config.get("resume_epoch", None)
    resume_override_lr = optimizer_config.get("resume_override_lr", False)
    resume_lr = optimizer_config.get("resume_lr", None)
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    encoder_cpu_offload = config.get("encoder_cpu_offload", False)
    use_context_parallel = config.get("use_context_parallel", False)
    use_skiparse_context_parallel = config.get("use_skiparse_context_parallel", False)
    deterministic_training = config.get("deterministic_training", False)

    # save config
    output_dir = config.get("output_dir", "./output_rl_lora")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

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

    # ===== FSDP mesh (same as train.py / train_osp_RL.py) =====
    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size:
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning: GPU nums not enough! FSDP size reset to {fsdp_size}!")
    elif world_size % fsdp_size != 0:
        raise ValueError(f"world_size % fsdp_size != 0, fsdp error!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    # ===== Context Parallelism (CP) init =====
    dp_group = dist.group.WORLD
    cp_size = 1
    use_context_parallel = use_context_parallel and config.get("cp_size", 1) > 1
    skiparse_cp_size = 1
    use_skiparse_context_parallel = use_skiparse_context_parallel and config.get("skiparse_cp_size", 1) > 1 and sparse_ratio > 1
    use_global_context_parallel = use_context_parallel or use_skiparse_context_parallel
    global_cp_size = 1
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
        log_on_main_process(logger, f"Using context parallel: global_cp_size={global_cp_size}, cp_size={cp_size}, skiparse_cp_size={skiparse_cp_size}")
        cp_state.reset(global_cp_group=global_cp_group, cp_group=cp_group, skiparse_cp_group=skiparse_cp_group, full_cp_group=full_cp_group)

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
    text_encoder_device_mesh = None
    if text_encoder_config.get("use_fsdp", False):
        num_replicate = max(world_size // 8, 1)
        num_shard = world_size // num_replicate
        text_encoder_device_mesh = init_device_mesh("cuda", (num_replicate, num_shard), mesh_dim_names=("replicate", "shard"))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False),
        device_mesh=text_encoder_device_mesh,
    )
    log_on_main_process(logger, f"Text encoder initialized, memory: {get_memory_allocated()} GiB")

    text_encoder_use_fsdp = text_encoder_config.get("use_fsdp", False)
    if encoder_cpu_offload:
        log_on_main_process(logger, "Offloading VAE and text encoder to CPU to save GPU memory...")
        vae.model.to("cpu")
        if not text_encoder_use_fsdp:
            text_encoder.model.to("cpu")
        torch.cuda.empty_cache()
        log_on_main_process(logger, f"After encoder CPU offload, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing scheduler...")
    scheduler = schedulers[scheduler_config.get("scheduler_name", "flow_matching")](**scheduler_config)

    # ========== Init Diffusion Model + LoRA + FSDP2 ==========
    log_on_main_process(logger, "Initializing diffusion model...")
    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False

    # Step 1: Load pretrained base model
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        # Base model checkpoint file: load weights BEFORE LoRA/FSDP2 wrapping
        # so that key names match the original model structure
        log_on_main_process(logger, f"Load base model from checkpoint file {pretrained_model_dir_or_checkpoint}")
        model = models[model_name](**model_config)
        if pretrained_model_dir_or_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load
            full_sd = safe_load(pretrained_model_dir_or_checkpoint, device="cpu")
        else:
            full_sd = torch.load(pretrained_model_dir_or_checkpoint, mmap=True, weights_only=True, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(full_sd, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"[Base model checkpoint] missing_keys: {missing_keys}")
            if unexpected_keys:
                print(f"[Base model checkpoint] unexpected_keys: {unexpected_keys}")
        del full_sd
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

    # Step 2: Apply LoRA (before FSDP2 wrapping)
    # NOTE: Do NOT freeze base model before FSDP2 wrap!
    # FSDP2's fully_shard packs module parameters into FlatParameters.
    # If base params have requires_grad=False before wrapping, the FlatParameter
    # will also be requires_grad=False, breaking gradient tracking through LoRA layers.
    # Instead, we keep all params requires_grad=True for FSDP2, and only pass
    # LoRA params to the optimizer (Step 8) so only LoRA gets updated.
    if lora_path:
        log_on_main_process(logger, f"Loading existing LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model.set_adapter("default")
    else:
        log_on_main_process(logger, f"Initializing new LoRA with rank={lora_rank}, alpha={lora_alpha}")
        log_on_main_process(logger, f"LoRA target modules: {lora_target_modules}")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, peft_config)

    if rank == 0:
        model.print_trainable_parameters()

    # CRITICAL: After PEFT's get_peft_model / PeftModel.from_pretrained, base params
    # are requires_grad=False. We must re-enable requires_grad for ALL params before
    # FSDP2 wrapping, because FSDP2's fully_shard creates FlatParameters that inherit
    # requires_grad from the original params. If base params are frozen, the FlatParameter
    # (which contains both base + LoRA params in the same module) may lose gradient
    # tracking, causing loss.backward() to fail with "does not require grad".
    # Only LoRA params will be passed to the optimizer (Step 8) so base weights
    # still won't be updated.
    model.requires_grad_(True)

    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    model.train()

    if model_cpu_offload:
        log_on_main_process(logger, "Moving model to CPU for FSDP CPU offloading to prevent NPU OOM...")
        model.to("cpu")
        torch.cuda.empty_cache()

    # Step 3: FSDP2 wrap (wraps the entire PeftModel including LoRA params)
    # All params are requires_grad=True at this point so FSDP2 FlatParameters
    # maintain gradient tracking through the forward pass.
    log_on_main_process(logger, "Starting FSDP2 wrapping...")
    import sys; sys.stdout.flush(); sys.stderr.flush()
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
    log_on_main_process(logger, "FSDP2 wrapping completed successfully.")
    sys.stdout.flush(); sys.stderr.flush()

    if not has_loaded_pretrained_model:
        init_device = "cpu" if model_cpu_offload else device
        model.to_empty(device=init_device)
        set_seed(seed, device_specific=False)
        base_model.reset_parameters()

    log_on_main_process(logger, f"Diffusion model (LoRA + FSDP2) initialized, memory: {get_memory_allocated()} GiB")
    sys.stdout.flush(); sys.stderr.flush()

    # Step 5: Gradient checkpointing
    if gradient_checkpointing:
        log_on_main_process(logger, "Using gradient checkpointing")
        if hasattr(base_model, 'set_gradient_checkpointing'):
            base_model.set_gradient_checkpointing(True)
        elif hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()

    # Step 5.5: Validate disable_adapter() works with FSDP2
    if (kl_reward > 0 or kl_beta > 0) and hasattr(model, 'disable_adapter'):
        try:
            with model.disable_adapter():
                log_on_main_process(logger, "disable_adapter() context manager works under FSDP2.")
        except Exception as e:
            log_on_main_process(logger, f"WARNING: disable_adapter() failed under FSDP2: {e}")
            log_on_main_process(logger, "KL computation may not work correctly. Consider using a separate ref_model.")

    # Step 6: EMA (FSDP-aware EMA, same as train_osp_RL.py)
    log_on_main_process(logger, "Initializing EMA model...")
    ema_model = EMAModel(model, decay=ema_decay, update_interval=ema_update_interval)
    _lora_ema_keys = {n for n, _ in model.named_parameters() if 'lora_' in n}
    _orig_ema_update = ema_model.update

    @torch.no_grad()
    def _lora_only_ema_update(model, step):
        if step % ema_model.update_interval != 0:
            return
        for name, param in model.named_parameters():
            shadow_param = ema_model.shadow_params[name]
            if name in _lora_ema_keys:
                shadow_param.data.sub_(
                    ema_model.one_minus_decay * (shadow_param.data.float() - param.data.float())
                )
            else:
                shadow_param.data.copy_(param.data)

    ema_model.update = _lora_only_ema_update
    log_on_main_process(logger, f"EMA model initialized (LoRA-only update, {len(_lora_ema_keys)} LoRA keys), memory: {get_memory_allocated()} GiB")

    # Step 7: Checkpointer
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
    # NOTE: base model checkpoint file loading has been moved to Step 1 (before LoRA/FSDP2 wrapping)
    # to avoid key name mismatch between raw checkpoint keys and PEFT-wrapped model keys.

    if not has_loaded_pretrained_model:
        log_on_main_process(logger, f"Warning! Training from scratch, pretrained_model_dir_or_checkpoint={pretrained_model_dir_or_checkpoint}")

    # Step 8: Optimizer (only LoRA params)
    # Since we keep all params requires_grad=True for FSDP2 compatibility,
    # we identify LoRA params by name (lora_A, lora_B) to pass to the optimizer.
    log_on_main_process(logger, "Initializing optimizer...")
    learning_rate = optimizer_config.get("lr", 5e-4)
    weight_decay_val = optimizer_config.get("weight_decay", 1e-2)
    lora_param_names = {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}
    trainable_parameters = [
        p for n, p in model.named_parameters()
        if any(lora_key in n for lora_key in lora_param_names)
    ]
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        lora_params = sum(p.numel() for p in trainable_parameters)
        log_on_main_process(logger, f"Optimizer: {lora_params:,} LoRA params / {total_params:,} total params")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
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
        override_lr = None
        if resume_lr is not None:
            override_lr = resume_lr
        elif resume_override_lr:
            override_lr = learning_rate
        if override_lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = override_lr
            log_on_main_process(
                logger,
                f"Resume LR override enabled, force optimizer lr to {override_lr}",
            )

    resume_global_step = 0 if checkpointer.last_training_iteration is None else checkpointer.last_training_iteration
    start_epoch = 0
    if checkpointer.last_training_iteration is not None:
        if resume_epoch is not None:
            start_epoch = int(resume_epoch)
            log_on_main_process(
                logger,
                f"Resume epoch manually specified as {start_epoch}. "
                "Skipping auto epoch recovery from rl_training_state.json.",
            )
        else:
            rl_training_state = load_rl_training_state(output_dir, resume_global_step)
            if rl_training_state is not None:
                start_epoch = int(rl_training_state.get("next_epoch", 0))
                saved_global_step = int(rl_training_state.get("global_step", resume_global_step))
                if saved_global_step != resume_global_step:
                    log_on_main_process(
                        logger,
                        f"Warning! RL training state global_step={saved_global_step} "
                        f"does not match checkpoint folder step={resume_global_step}. "
                        f"Using checkpoint folder step.",
                    )
            else:
                log_on_main_process(
                    logger,
                    "Warning! RL training state file not found for resume checkpoint. "
                    "Falling back to start_epoch=0; epoch continuity cannot be guaranteed. "
                    "Set config.resume_epoch to recover the expected epoch manually.",
                )
    elif resume_epoch is not None:
        start_epoch = int(resume_epoch)
        log_on_main_process(
            logger,
            f"resume_epoch={start_epoch} is set but no checkpoint was found in output_dir. "
            "Training will start from scratch with the specified epoch index.",
        )

    set_seed(seed, device_specific=True, process_group=dp_group, deterministic=deterministic_training)

    # ========== RL Dataset & Reward ====================
    log_on_main_process(logger, f"Initializing reward functions with config: {reward_fn_config}")
    import sys
    try:
        print(f"[Rank {rank}] Importing ospnext.rewards.rewards ...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        import ospnext.rewards.rewards
        print(f"[Rank {rank}] Import successful, calling multi_score ...", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        reward_fn = getattr(ospnext.rewards.rewards, 'multi_score')(device, reward_fn_config)
        print(f"[Rank {rank}] multi_score initialized successfully.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        log_on_main_process(logger, f"ERROR: Failed to import ospnext.rewards.rewards: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        def reward_fn(videos, prompts, metadata, only_strict=True):
            B = len(prompts)
            return {"avg": np.ones(B, dtype=np.float32)}, {}
    # Synchronize all ranks after reward init to catch early failures
    dist.barrier()
    log_on_main_process(logger, "All ranks passed reward initialization.")

    # Prompt dataset
    text_tokenizer_path = data_config.get("dataset_config", {}).get("text_tokenizer_path", None)
    text_max_length = data_config.get("dataset_config", {}).get("tokenizer_max_length", text_encoder_config.get("text_len", 512))
    if text_tokenizer_path is None:
        raise ValueError("data_config.dataset_config.text_tokenizer_path must be specified.")
    if prompt_file is None:
        raise ValueError("prompt_file must be specified for RL training.")

    train_dataset = TextPromptDataset(
        file_path=prompt_file,
        text_tokenizer_path=text_tokenizer_path,
        text_max_length=text_max_length,
    )

    # dp_size and dp_rank
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

    # # Eval dataset
    # test_dataloader = None
    # if eval_prompt_file is not None:
    #     eval_dataset = TextPromptDataset(
    #         file_path=eval_prompt_file,
    #         text_tokenizer_path=text_tokenizer_path,
    #         text_max_length=text_max_length,
    #     )
    #     test_dataloader = DataLoader(
    #         eval_dataset,
    #         batch_size=sample_batch_size,
    #         collate_fn=TextPromptDataset.collate_fn,
    #         shuffle=False,
    #         num_workers=4,
    #     )

    # Eval dataset
    test_dataloader = None
    eval_sampler = None
    ddp_rank_for_eval, fsdp_local_rank, ddp_size_for_eval = get_ddp_rank_and_fsdp_local_rank(
        rank=rank,
        fsdp_size=fsdp_size,
        world_size=world_size,
    )

    if eval_prompt_file is not None:
        eval_dataset = TextPromptDataset(
            file_path=eval_prompt_file,
            text_tokenizer_path=text_tokenizer_path,
            text_max_length=text_max_length,
        )

        # Important:
        # - FSDP ranks inside the same replica must see the same eval batches
        # - Different DDP replicas should see different eval subsets
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=ddp_size_for_eval,
            rank=ddp_rank_for_eval,
            shuffle=False,
            drop_last=False,
        )

        test_dataloader = DataLoader(
            eval_dataset,
            batch_size=sample_batch_size,
            sampler=eval_sampler,
            collate_fn=TextPromptDataset.collate_fn,
            num_workers=4,
            pin_memory=True,
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
    # 仅对 SDE 步进行训练，timestep_fraction 在 SDE 步范围内进一步控制训练比例
    num_train_timesteps = int(sde_steps * timestep_fraction)
    train_timestep_indices = list(range(num_train_timesteps))

    # Sigma schedule
    sigmas_schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas_schedule = shift * sigmas_schedule / (1 + (shift - 1) * sigmas_schedule)

    # Executor for async reward computation
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ========== Logging ==========
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log_on_main_process(logger, f"""
    {'=' * 20}Start RL Training (GRPO + LoRA + FSDP2){'=' * 20}
    Model: {model_name}
    LoRA rank: {lora_rank}, alpha: {lora_alpha}
    LoRA target modules: {lora_target_modules}
    Trainable parameters: {params_nums_to_str(trainable_params)} / {params_nums_to_str(total_params)}
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
    Resume global step: {resume_global_step}
    Start epoch: {start_epoch}
    Resume epoch override: {resume_epoch}
    Resume LR override: {resume_override_lr}
    Resume LR value: {resume_lr if resume_lr is not None else 'follow optimizer_config.lr'}
    Current optimizer lr: {optimizer.param_groups[0]['lr']}
    Gradient accumulation steps: {gradient_accumulation_steps}
    FSDP size: {fsdp_size}
    DDP size: {ddp_size}
    World size: {world_size}
    dp_size: {dp_size}
    cp_size: {cp_size}
    skiparse_cp_size: {skiparse_cp_size}
    global_cp_size: {global_cp_size}
    Use Context Parallel: {use_context_parallel}
    Use Skiparse Context Parallel: {use_skiparse_context_parallel}
    Use Full Blocks Context Parallel: {use_full_blocks_context_parallel}
    Reshard after forward: {reshard_after_forward}
    Model CPU offload: {model_cpu_offload}
    Video: {video_num_frames}f x {video_height}h x {video_width}w
    Output dir: {output_dir}
    {'=' * 20}{'=' * len('Start RL Training (GRPO + LoRA + FSDP2)')}{'=' * 20}
    """)

    # ========== Training Loop ==========
    global_step = resume_global_step
    last_completed_epoch = start_epoch - 1
    train_iter = iter(train_dataloader)

    # Compute latent shape
    vae_temporal_factor = 4
    vae_spatial_factor = 8
    latent_T = (video_num_frames - 1) // vae_temporal_factor + 1
    latent_H = video_height // vae_spatial_factor
    latent_W = video_width // vae_spatial_factor
    latent_C = model_config.get("in_dim", 16)
    latent_shape = (sample_batch_size, latent_C, latent_T, latent_H, latent_W)

    log_on_main_process(logger, f"Latent shape: {latent_shape}")

    # Number of training timesteps determines gradient accumulation
    accum_steps_total = gradient_accumulation_steps * num_train_timesteps

    for epoch in range(start_epoch, num_epochs):
        # ==================== SAMPLING PHASE ====================
        model.eval()

        # ===== Sample 阶段禁用 CP / skiparse_cp，只用 DDP =====
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
            log_on_main_process(logger, "Sample phase: keeping skiparse_cp, disabling ulysses CP only")
            # 只禁用 Ulysses CP，保留 skiparse_cp 用于采样阶段
            if use_context_parallel:
                # 清除 Ulysses CP 相关状态
                cp_state.cp_group = None
                cp_state.cp_rank = 0
                cp_state.cp_size = 1
                # global_cp 退化为 skiparse_cp
                cp_state.global_cp_group = cp_state.skiparse_cp_group
                cp_state.global_cp_rank = cp_state.skiparse_cp_rank
                cp_state.global_cp_size = cp_state.skiparse_cp_size
                # full blocks cp 退化为无 CP（只有 skiparse blocks 用 skiparse_cp）
                cp_state.full_cp_group = None
                cp_state.full_cp_rank = 0
                cp_state.full_cp_size = 1
            # skiparse_cp 状态保持不变
            _cp_state_module.USE_CONTEXT_PARALLEL = None
            _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = None
            _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = None
            base_model_inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
            if hasattr(base_model_inner, 'rope_wrapper') and base_model_inner.rope_wrapper is not None:
                base_model_inner.rope_wrapper.cache.clear()
            if hasattr(base_model_inner, 'mask_preprocessor'):
                base_model_inner.mask_preprocessor.cache.clear()
            if hasattr(base_model_inner, 'context_preprocessor'):
                base_model_inner.context_preprocessor.shard_seq_lens_cache.clear()
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # 采样阶段启用 reshard_after_forward=True，避免 FSDP 参数一直保持 unsharded
        if reshard_after_forward is not None and not reshard_after_forward:
            model.set_reshard_after_forward(True, recurse=True)

        samples = []
        all_prompts = []
        last_videos_cpu = None
        last_prompts = None

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

            if encoder_cpu_offload:
                vae.model.to(device)
                if not text_encoder_use_fsdp:
                    text_encoder.model.to(device)

            # Encode prompts
            with torch.no_grad():
                text_embeddings = text_encoder(prompt_ids, prompt_mask)
            torch.cuda.synchronize()

            if encoder_cpu_offload:
                vae.model.to("cpu")
                if not text_encoder_use_fsdp:
                    text_encoder.model.to("cpu")
                torch.cuda.empty_cache()

            # Save/eval checkpoint
            # if batch_idx == 0 and epoch % save_interval == 0 and epoch > 0:
            #     log_on_main_process(logger, f"Saving checkpoint at epoch {epoch}...")
            #     checkpointer.save(model, optimizer, None, epoch)
            #     ema_model.store(model)
            #     ema_model.ema_copy_to_model(model)
            #     checkpointer.save_ema_model(model, epoch)
            #     ema_model.restore(model)
            #     adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{epoch:09d}")
            #     # Also save LoRA-specific checkpoint
            #     if hasattr(model, 'save_pretrained'):
            #         save_lora_checkpoint(model, output_dir, epoch)
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # Skip first 2 epochs (collecting group statistics)
            # if epoch < 2:
            #     continue

            # 清理显存
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Sample
            for sample_t in range(sample_time_per_prompt):
                with torch.no_grad():
                    # 采样阶段使用 skiparse_cp：传入 skiparse_cp_group 同步初始噪声
                    _sample_cp_group = skiparse_cp_group if use_skiparse_context_parallel else None
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
                        cp_group=_sample_cp_group,
                        sde_steps=sde_steps,
                    )

                # Stack latents and log_probs（仅包含 SDE 步）
                latents_stacked = torch.stack(latents_list, dim=1)
                log_probs_stacked = torch.stack(log_probs_list, dim=1)
                kl_stacked = torch.stack(kl_list, dim=1)
                del latents_list, log_probs_list, kl_list

                # timesteps 仅对应 SDE 步的索引
                timesteps_repeated = torch.arange(sde_steps, device=device).unsqueeze(0).expand(
                    sample_batch_size, -1
                )

                # Compute rewards asynchronously
                videos_cpu = videos.detach().cpu()
                del videos
                videos_for_reward = (videos_cpu.float() + 1.0) / 2.0
                rewards_future = executor.submit(
                    reward_fn, videos_for_reward.numpy(), prompts, prompt_metadata, True
                )
                del videos_for_reward

                # Save last batch for wandb
                last_videos_cpu = videos_cpu
                last_prompts = list(prompts)

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
                del latents_stacked, log_probs_stacked, kl_stacked, videos_cpu
                torch.cuda.empty_cache()

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
            _cp_state_module.USE_CONTEXT_PARALLEL = _saved_USE_CP
            _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = _saved_USE_SKIPARSE_CP
            _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = _saved_USE_FULL_CP
            base_model_inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
            if hasattr(base_model_inner, 'rope_wrapper') and base_model_inner.rope_wrapper is not None:
                base_model_inner.rope_wrapper.cache.clear()
            if hasattr(base_model_inner, 'mask_preprocessor'):
                base_model_inner.mask_preprocessor.cache.clear()
            if hasattr(base_model_inner, 'context_preprocessor'):
                base_model_inner.context_preprocessor.shard_seq_lens_cache.clear()
            dist.barrier()
            torch.cuda.synchronize()

        # if epoch < 2:
        #     continue

        # Wait for all rewards
        for sample in tqdm(samples, desc="Waiting for rewards", disable=(rank != 0)):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        # Collate samples
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # Log videos
        if epoch % 1 == 0 and rank == 0 and last_videos_cpu is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_vis = min(8, len(last_videos_cpu))
                sample_indices = random.sample(range(len(last_videos_cpu)), num_vis)
                for idx, i in enumerate(sample_indices):
                    video = last_videos_cpu[i].numpy().transpose(1, 2, 3, 0)
                    frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                    imageio.mimsave(os.path.join(tmpdir, f"{idx}.mp4"), frames, fps=16, codec="libx264", format='FFMPEG')

                if wandb.run is not None:
                    sampled_prompts = [last_prompts[i] for i in sample_indices]
                    wandb.log(
                        {"videos": [
                            wandb.Video(os.path.join(tmpdir, f"{idx}.mp4"), caption=f"{p:.100}", fps=16)
                            for idx, p in enumerate(sampled_prompts)
                        ]},
                        step=global_step,
                    )

        # Apply KL penalty to rewards
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        kl_on_device = samples["kl"].to(device)
        num_steps_dim = kl_on_device.shape[1] if kl_on_device.dim() > 1 else sde_steps
        avg_expanded = samples["rewards"]["avg"].unsqueeze(-1).expand(-1, num_steps_dim)
        samples["rewards"]["avg"] = avg_expanded - kl_reward * kl_on_device
        del kl_on_device

        # Gather rewards across dp ranks
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
                # value is a numpy array
                if '_strict_accuracy' not in key and '_accuracy' not in key:
                    log_dict[f"reward_{key}"] = float(value.mean())
                    log_dict[f"reward/{key}_mean"] = float(value.mean())
                    log_dict[f"reward/{key}_std"] = float(value.std())
                    log_dict[f"reward/{key}_abs_mean"] = float(np.abs(value).mean())
                    log_dict[f"reward/{key}_max"] = float(value.max())
                    log_dict[f"reward/{key}_min"] = float(value.min())

            if wandb.run is not None:
                wandb.log(log_dict, step=global_step)

        # Compute advantages
        if per_prompt_stat_tracking and stat_tracker is not None:
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

        # Redistribute advantages
        advantages = torch.as_tensor(advantages).float()

        # Log advantage statistics (全局 advantages，redistribute 之前)
        if rank == 0:
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            adv_abs_mean = advantages.abs().mean().item()
            adv_max = advantages.max().item()
            adv_min = advantages.min().item()
            log_on_main_process(logger, f"Epoch {epoch} | advantage mean: {adv_mean:.4f} | advantage std: {adv_std:.4f} | advantage abs mean: {adv_abs_mean:.4f} | advantage max: {adv_max:.4f} | advantage min: {adv_min:.4f}")
            if wandb.run is not None:
                wandb.log({
                    "advantage/mean": adv_mean,
                    "advantage/std": adv_std,
                    "advantage/abs_mean": adv_abs_mean,
                    "advantage/max": adv_max,
                    "advantage/min": adv_min,
                }, step=global_step)

        if advantages.dim() == 1:
            local_advantages = advantages.reshape(dp_size, -1)[dp_rank]
            local_advantages = local_advantages.unsqueeze(-1).expand(-1, num_train_timesteps).contiguous()
        else:
            local_advantages = advantages.reshape(dp_size, -1, *advantages.shape[1:])[dp_rank]
        samples["advantages"] = local_advantages

        if rank == 0:
            log_on_main_process(logger, f"Epoch {epoch} | local advantages abs mean: {samples['advantages'].abs().mean().item():.4f} | kl mean: {samples['kl'].mean().item():.4f}")

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
        backward_counter = 0

        for inner_epoch in range(num_inner_epochs):
            model.train()
            # 训练阶段恢复原始 reshard_after_forward 设置
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(reshard_after_forward, recurse=True)

            # 直接使用采样阶段保存的 log_probs 作为 old policy 基准。
            # 不做 RECOMPUTE：采样阶段的 log_probs 由当时的 LoRA model 生成，
            # 随着训练推进 LoRA 权重更新，训练阶段重新计算的 log_prob 与采样时的
            # log_probs 会逐渐产生差异，使 ratio 偏离 1.0，PPO clip 机制生效。

            # Shuffle along batch dimension (samples are on CPU)
            perm = torch.randperm(total_batch_size_local, device=device)
            # 同步 perm 索引：只要同一 CP 组内的卡取数据的顺序完全一致，就能保证整个 micro_batch 的输入一致，避免了广播巨大的 latents 导致通信阻塞
            if use_global_context_parallel:
                torch.distributed.broadcast(perm, group_src=0, group=global_cp_group)
            perm = perm.cpu()
            
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
            # latents[:, j] 将不再对应 timestep j，导致 sigma 和 latent 错配。

            # Split into micro-batches
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

                        # KL regularization against ref model (LoRA disabled)
                        if kl_beta > 0:
                            with torch.no_grad():
                                with model.disable_adapter():
                                    _, _, ref_prev_sample_mean, ref_std_dev_t, ref_dt = compute_log_prob_for_training(
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
                    unclipped_loss = -adv * ratio
                    clipped_loss = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                    if kl_beta > 0:
                        kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * (std_dev_t * ref_dt) ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + kl_beta * kl_loss
                        info["kl_loss"].append(kl_loss.detach())
                    else:
                        loss = policy_loss

                    # Scale loss for gradient accumulation
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

                    # Optimizer step
                    if backward_counter % accum_steps_total == 0:
                        # --- DEBUG: check LoRA gradient before optimizer step ---
                        if rank == 0 and global_step < 3:
                            for _dn, _dp in model.named_parameters():
                                if 'lora_' in _dn:
                                    _local = _dp.grad._local_tensor if isinstance(_dp.grad, DTensor) else (_dp.grad if _dp.grad is not None else None)
                                    if _local is not None:
                                        print(f"[DEBUG grad] {_dn}: grad_norm={_local.norm().item():.8f}, "
                                              f"param_norm={(_dp._local_tensor if isinstance(_dp, DTensor) else _dp).norm().item():.8f}")
                                    else:
                                        print(f"[DEBUG grad] {_dn}: grad is None!")
                                    break
                        # --- END DEBUG ---
                        grad_norm = adaptive_grad_clipper.adaptive_clip(trainable_parameters)
                        optimizer.step()
                        model.zero_grad(set_to_none=True)
                        # --- DEBUG: check LoRA param after optimizer step ---
                        if rank == 0 and global_step < 3:
                            for _dn, _dp in model.named_parameters():
                                if 'lora_' in _dn:
                                    _local = _dp._local_tensor if isinstance(_dp, DTensor) else _dp
                                    print(f"[DEBUG post-step] {_dn}: param_norm={_local.norm().item():.8f}")
                                    break
                        # --- END DEBUG ---
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

            # Handle remaining gradients
            if backward_counter % accum_steps_total != 0:
                grad_norm = adaptive_grad_clipper.adaptive_clip(trainable_parameters)
                optimizer.step()
                model.zero_grad(set_to_none=True)
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

        # 训练阶段结束后同步并清理
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
            save_rl_training_state(output_dir, global_step, epoch + 1)
            # Also save LoRA-specific checkpoint
            if hasattr(model, 'save_pretrained'):
                save_lora_checkpoint(model, output_dir, global_step)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        last_completed_epoch = epoch

        # ==================== EVAL PHASE ====================
        if (test_dataloader is not None
                and global_step > 0
                and global_step % eval_freq == 0):
            log_on_main_process(logger, f"[Eval] Running eval video generation at global_step {global_step}...")
            model.eval()

            # 切换到 EMA 权重
            ema_model.store(model)
            ema_model.ema_copy_to_model(model)

            # Eval 阶段同样需要 reshard_after_forward=True
            if reshard_after_forward is not None and not reshard_after_forward:
                model.set_reshard_after_forward(True, recurse=True)

            # 禁用 CP（与采样阶段一致：仅禁用 Ulysses CP，保留 skiparse_cp）
            if use_global_context_parallel:
                log_on_main_process(logger, "[Eval] keeping skiparse_cp, disabling ulysses CP only")
                if use_context_parallel:
                    cp_state.cp_group = None
                    cp_state.cp_rank = 0
                    cp_state.cp_size = 1
                    cp_state.global_cp_group = cp_state.skiparse_cp_group
                    cp_state.global_cp_rank = cp_state.skiparse_cp_rank
                    cp_state.global_cp_size = cp_state.skiparse_cp_size
                    cp_state.full_cp_group = None
                    cp_state.full_cp_rank = 0
                    cp_state.full_cp_size = 1
                _cp_state_module.USE_CONTEXT_PARALLEL = None
                _cp_state_module.USE_SKIPARSE_CONTEXT_PARALLEL = None
                _cp_state_module.USE_FULL_BLOCKS_CONTEXT_PARALLEL = None
                base_model_inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
                if hasattr(base_model_inner, 'rope_wrapper') and base_model_inner.rope_wrapper is not None:
                    base_model_inner.rope_wrapper.cache.clear()
                if hasattr(base_model_inner, 'mask_preprocessor'):
                    base_model_inner.mask_preprocessor.cache.clear()
                if hasattr(base_model_inner, 'context_preprocessor'):
                    base_model_inner.context_preprocessor.shard_seq_lens_cache.clear()
                dist.barrier()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            eval_videos_cpu = []
            eval_prompts = []

            for eval_batch in tqdm(test_dataloader, desc=f"[Eval] Generating videos", disable=(rank != 0)):
                eval_prompt_texts = eval_batch[PROMPT]
                eval_prompt_ids = eval_batch[PROMPT_IDS].to(device)
                eval_prompt_mask = eval_batch[PROMPT_MASK].to(device)

                if encoder_cpu_offload:
                    vae.model.to(device)
                    if not text_encoder_use_fsdp:
                        text_encoder.model.to(device)

                with torch.no_grad():
                    eval_text_embeddings = text_encoder(eval_prompt_ids, eval_prompt_mask)
                torch.cuda.synchronize()

                if encoder_cpu_offload:
                    vae.model.to("cpu")
                    if not text_encoder_use_fsdp:
                        text_encoder.model.to("cpu")
                    torch.cuda.empty_cache()

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
                base_model_inner = model.get_base_model() if hasattr(model, 'get_base_model') else model
                if hasattr(base_model_inner, 'rope_wrapper') and base_model_inner.rope_wrapper is not None:
                    base_model_inner.rope_wrapper.cache.clear()
                if hasattr(base_model_inner, 'mask_preprocessor'):
                    base_model_inner.mask_preprocessor.cache.clear()
                if hasattr(base_model_inner, 'context_preprocessor'):
                    base_model_inner.context_preprocessor.shard_seq_lens_cache.clear()
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
                        video = eval_all_videos[idx].numpy().transpose(1, 2, 3, 0)
                        frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video]
                        imageio.mimsave(
                            os.path.join(tmpdir, f"eval_{idx}.mp4"),
                            frames, fps=16, codec="libx264", format='FFMPEG',
                        )

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "eval/videos": [
                                    wandb.Video(
                                        os.path.join(tmpdir, f"eval_{idx}.mp4"),
                                        caption=f"{eval_prompts[idx]:.100}",
                                        fps=16,
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
    completed_epochs = max(start_epoch, last_completed_epoch + 1)
    log_on_main_process(logger, f"Saving final checkpoint at global_step {global_step}...")
    checkpointer.save(model, optimizer, None, global_step)
    ema_model.store(model)
    ema_model.ema_copy_to_model(model)
    checkpointer.save_ema_model(model, global_step)
    ema_model.restore(model)
    adaptive_grad_clipper.save(output_dir=f"{output_dir}/{checkpoint_prefix}{global_step:09d}")
    save_rl_training_state(output_dir, global_step, completed_epochs)
    # Also save LoRA-specific checkpoint
    if hasattr(model, 'save_pretrained'):
        save_lora_checkpoint(model, output_dir, global_step)

    log_on_main_process(logger, f"""
    {'=' * 20}End RL Training (LoRA + FSDP2){'=' * 20}
    Total epochs: {completed_epochs}
    Total global steps: {global_step}
    Model saved to {output_dir}
    {'=' * 20}{'=' * len('End RL Training (LoRA + FSDP2)')}{'=' * 20}
    """)
    cleanup_distributed_env()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/gpu/osp_14b_RL_lora.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} does not exist!")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(config)
