import torch
import os
from ospnext.modules.osp_next import OSPNextModel

def load_lora_and_merge(
    model,
    lora_path,
    lora_rank=32,
    lora_alpha=64,
    lora_target_modules=None,
    logger=None,
    rank=0,
):
    """
    Load LoRA weights from a manually saved adapter_model.bin, then merge into base model.
    """
    from peft import LoraConfig, get_peft_model
    if lora_target_modules is None:
        lora_target_modules = [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        ]

    if not os.path.isfile(lora_path):
        raise ValueError(f"LoRA file not found: {lora_path}")

    if logger is not None:
        from ospnext.utils.log_utils import log_on_main_process
        log_on_main_process(logger, f"Loading LoRA from {lora_path}")
        log_on_main_process(logger, f"LoRA rank={lora_rank}, alpha={lora_alpha}")
        log_on_main_process(logger, f"LoRA target_modules={lora_target_modules}")

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.set_adapter("default")

    lora_sd = torch.load(lora_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_sd, strict=False)

    if rank == 0:
        missing_lora = [k for k in missing if "lora_" in k]
        if missing_lora:
            print(f"[LoRA] missing lora keys: {len(missing_lora)}, example: {missing_lora[:5]}")
        print(f"[LoRA] unexpected keys: {len(unexpected)}")

    # sanity check
    missing_lora = [k for k in missing if "lora_" in k]
    if len(unexpected) > 0:
        raise RuntimeError(f"LoRA load has unexpected keys, example: {unexpected[:20]}")
    if len(missing_lora) > 0:
        raise RuntimeError(f"LoRA load missing LoRA keys, example: {missing_lora[:20]}")

    if logger is not None:
        log_on_main_process(logger, "LoRA weights loaded successfully, merging into base model...")

    model = model.merge_and_unload()

    if logger is not None:
        log_on_main_process(logger, "LoRA merged into base model successfully.")

    return model

model_path = "/path/to/osp_next_model"
lora_path = "/path/to/lora_model"
lora_target_modules = ['cross_attn.k', 'self_attn.o', 'self_attn.q', 'cross_attn.o', 'cross_attn.v', 'cross_attn.q', 'self_attn.k', 'self_attn.v']
lora_rank = 32
lora_alpha = 64
save_path = "/path/to/merged_model"

model = OSPNextModel.from_pretrained(model_path)
model = load_lora_and_merge(
    model=model,
    lora_path=lora_path,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_target_modules=lora_target_modules,
    logger=None,
    rank=0,
)

model.save_pretrained(save_path)