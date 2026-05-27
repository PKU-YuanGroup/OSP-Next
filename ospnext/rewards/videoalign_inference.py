"""
VideoAlign Inference

Consolidated from VideoAlign/inference.py.
All imports now use local modules within ospnext/rewards/ instead of
requiring the VideoAlign source directory on sys.path.
"""

import json
import os
from collections.abc import Mapping

import torch

from ospnext.rewards.videoalign_vision_process import process_vision_info
from ospnext.rewards.videoalign_prompt_template import build_prompt
from ospnext.rewards.videoalign_model import (
    DataConfig,
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    load_model_from_checkpoint,
    create_model_and_processor,
)


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return (
        config_dict["data_config"],
        None,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict["inference_config"] if "inference_config" in config_dict else None,
    )


class VideoVLMRewardInference:
    def __init__(self, load_from_pretrained, load_from_pretrained_step=-1, device='cuda', dtype=torch.bfloat16):
        import sys
        def _log(msg):
            """Force-flush log to both stdout and stderr to survive NPU crashes."""
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

        _log(f"[VideoVLMRewardInference] __init__ called: pretrained={load_from_pretrained}, device={device}")

        config_path = os.path.join(load_from_pretrained, "model_config.json")
        _log(f"[VideoVLMRewardInference] Loading config from {config_path}")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        # ====================================================================
        # Strategy: Load model on CPU in float32, merge LoRA, then cast dtype
        # and move to NPU. This avoids potential NPU issues with PEFT adapters.
        # ====================================================================
        # Override torch_dtype to float32 so that from_pretrained also loads in float32
        original_torch_dtype = model_config.torch_dtype
        model_config.torch_dtype = "float32"
        _log(f"[VideoVLMRewardInference] Step 1: Creating model on CPU in float32 (original dtype config: {original_torch_dtype})")
        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=True,  # Use SDPA by default to avoid flash_attn compatibility issues on NPU
            bf16=False,   # Do NOT cast to bf16 yet — merge LoRA in float32 first
            fp16=False,   # Do NOT cast to fp16 yet
            output_dir="",
        )
        
        _log(f"[VideoVLMRewardInference] Creating model from {model_config.model_name_or_path}")
        _log(f"[VideoVLMRewardInference] LoRA enabled: {peft_lora_config.lora_enable}")
        
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )
        _log("[VideoVLMRewardInference] Model and processor created successfully (on CPU, float32)")

        self.device = device
        self._target_dtype = dtype

        _log(f"[VideoVLMRewardInference] Step 2: Loading checkpoint on CPU")
        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        _log(f"[VideoVLMRewardInference] Loaded checkpoint step: {checkpoint_step}")

        # Step 3: Merge LoRA on CPU (float32) — safe and compatible
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                _log("[VideoVLMRewardInference] Step 3: Merging LoRA weights on CPU (float32)...")
                model = model.merge_and_unload()
                _log("[VideoVLMRewardInference] LoRA weights merged and adapter unloaded successfully")
            else:
                _log(f"[VideoVLMRewardInference] Model is not PeftModel (type={type(model).__name__}), skipping merge")
        except Exception as e:
            _log(f"[VideoVLMRewardInference] Warning: Failed to merge LoRA: {e}")
            import traceback
            traceback.print_exc()

        model.eval()

        # Step 4: Cast to target dtype on CPU first, then move to NPU/GPU
        _log(f"[VideoVLMRewardInference] Step 4: Casting model to {self._target_dtype} on CPU")
        model = model.to(dtype=self._target_dtype)
        _log(f"[VideoVLMRewardInference] Step 5: Moving merged model to {self.device}")
        try:
            model = model.to(device=self.device)
        except Exception as e:
            _log(f"[VideoVLMRewardInference] ERROR moving model to device: {e}")
            import traceback
            traceback.print_exc()
            raise

        self.model = model
        self.processor = processor
        _log(f"[VideoVLMRewardInference] Model ready on {self.device} with dtype {self._target_dtype}")

        self.data_config = data_config
        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward['VQ'] = (reward['VQ'] - self.inference_config['VQ_mean']) / self.inference_config['VQ_std']
            reward['MQ'] = (reward['MQ'] - self.inference_config['MQ_mean']) / self.inference_config['MQ_std']
            reward['TA'] = (reward['TA'] - self.inference_config['TA_mean']) / self.inference_config['TA_std']
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ] for video_path, prompt in zip(video_paths, prompts)
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            fps: float, sample rate of the videos.
            num_frames: int, number of frames of the videos.
            max_pixels: int, maximum pixels of the videos.
            use_norm: bool, whether to rescale the output rewards.
        Outputs:
            Rewards: List[dict], rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        rewards = [{'VQ': reward[0].item(), 'MQ': reward[1].item(), 'TA': reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]['Overall'] = rewards[i]['VQ'] + rewards[i]['MQ'] + rewards[i]['TA']

        return rewards
