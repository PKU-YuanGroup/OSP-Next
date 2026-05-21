import os
import math
import yaml
from argparse import ArgumentParser
import torch.distributed as dist

import torch
from ospnext.utils.utils import check_and_import_npu
check_and_import_npu()

from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoTokenizer

from ospnext.distributed.utils import (
    setup_distributed_env, 
    cleanup_distributed_env, 
    gather_tensor_list_to_one, 
    set_modules_to_forward_prefetch,
)
from ospnext.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
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
from ospnext.distributed.checkpoint import Checkpointer
from ospnext.utils.utils import str_to_precision, get_memory_allocated
from ospnext.utils.log_utils import get_logger, log_on_main_process
from ospnext.pipelines import pipelines
from ospnext.utils.infer_utils import load_prompts, load_images, save_videos, save_video_grid
from ospnext.utils.random_utils import set_seed

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
        # 只打印缺失的 lora_ 键，基础模型的键缺失是正常的，因为 lora 权重里本来就只有 lora_ 相关的键
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

def main(config):
    logger = get_logger()

    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model_name", "osp_next")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})
    # skiparse相关
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_1d = model_config.get("skiparse_1d", False)
    skiparse_2d = model_config.get("skiparse_2d", False)
    num_full_blocks = model_config.get("num_full_blocks", 0)

    # inference config
    pipeline_name = config.get("pipeline_name", "t2v")
    weight_dtype = config.get("weight_dtype", "bfloat16")
    prompt_txt = config.get("prompt_txt", None)
    batch_size = config.get("batch_size", 1)
    num_frames = config.get("num_frames", 49)
    height = config.get("height", 480)
    width = config.get("width", 832)
    save_fps = config.get("save_fps", 16)
    use_context_parallel = config.get("use_context_parallel", False)
    use_skiparse_context_parallel = config.get("use_skiparse_context_parallel", False)
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    explicit_prefetching_num_blocks = config.get("explicit_prefetching_num_blocks", 0)

    # LoRA config
    lora_path = config.get("lora_path", None)
    lora_rank = config.get("lora_rank", 32)
    lora_alpha = config.get("lora_alpha", 64)
    lora_target_modules = config.get(
        "lora_target_modules",
        [
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
        ]
    )

    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # init fsdp config
    fsdp_size = config.get("fsdp_size", 8)
    if fsdp_size > world_size: 
        fsdp_size = world_size
        log_on_main_process(logger, f"Warning, GPU nums are not enough! FSDP size reset to {fsdp_size}!")
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    dp_group = dist.group.WORLD # use default world group
    # init cp mesh if use context parallel
    cp_size = 1
    use_context_parallel = use_context_parallel and config.get("cp_size", 1) > 1
    # skiparse cp
    skiparse_cp_size = 1
    use_skiparse_context_parallel = use_skiparse_context_parallel and config.get("skiparse_cp_size", 1) > 1 and sparse_ratio > 1
    use_global_context_parallel = use_context_parallel or use_skiparse_context_parallel
    global_cp_size = 1
    # full cp size
    # 用于混合full attn blocks和skiparse blocks时对于full blocks采用的cp group
    full_cp_size = 1
    use_full_blocks_context_parallel = use_global_context_parallel and (skiparse_1d or skiparse_2d) and num_full_blocks > 0
    # use_full_blocks_context_parallel = False
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
        # dp * skiparse_cp * cp = world_size
        # 先skiparse_cp后cp，因为skiparse_cp的通信量更小
        dp_global_cp_mesh = init_device_mesh("cuda", (world_size // global_cp_size, skiparse_cp_size, cp_size), mesh_dim_names=("dp", "skiparse_cp", "cp"))
        dp_group = dp_global_cp_mesh["dp"].get_group()
        global_cp_group = dp_global_cp_mesh["skiparse_cp", "cp"]._flatten().get_group()
        skiparse_cp_group = dp_global_cp_mesh["skiparse_cp"].get_group()
        # 初始化的时候full_blocks_cp_group和cp_group是同一个group
        full_cp_group = cp_group = dp_global_cp_mesh["cp"].get_group()
        log_on_main_process(logger, f"We use context parrallel, global_cp_size: {global_cp_size}, cp_size: {cp_size}, skiparse_cp_size: {skiparse_cp_size}")
        cp_state.reset(global_cp_group=global_cp_group, cp_group=cp_group, skiparse_cp_group=skiparse_cp_group, full_cp_group=full_cp_group)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f, indent=4)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.get("text_tokenizer_path", None))
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device, # when no fsdp, we init the text_encoder on device
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False), # when using fsdp, we shard the text encoder by ddp_fsdp mesh
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Init model from scratch")
        with torch.device("meta"):
            model = models[model_name](**model_config)
    else:
        raise ValueError(f"In inference mode, pretrained_model_dir_or_checkpoint {pretrained_model_dir_or_checkpoint} must be specified!")

    if use_context_parallel or use_full_blocks_context_parallel:
        if use_context_parallel and model.num_heads % cp_size != 0:
            raise ValueError(f"When using context parallel, num_heads {model.num_heads} mush be mutiple of cp_size {cp_size}!")
        if use_full_blocks_context_parallel:
            if global_cp_size <= model.num_heads and model.num_heads % global_cp_size == 0:
                full_cp_size = global_cp_size
            # 找到model.num_heads与global_cp_size的最大公约数
            else:
                gcd = math.gcd(model.num_heads, global_cp_size)
                full_cp_size = gcd
            # full_cp_size一定是global_cp_size的约数，而global_cp_size一定是world_size的约数
            # 所以full_cp_group一定是global_cp_group的子集
            dummy_mesh = init_device_mesh("cuda", (world_size // full_cp_size, full_cp_size), mesh_dim_names=("dummy", "full_cp"))
            full_cp_group = dummy_mesh["full_cp"].get_group()
            cp_state.reset(full_cp_group=full_cp_group)
    
    model.eval()

    # if model was initialized on meta, materialize and load base checkpoint first
    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(seed, device_specific=False) # for init
        model.reset_parameters() # we should call reset_parameters because we init model at meta device 

    if pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_checkpoint {pretrained_model_dir_or_checkpoint}")
        if pretrained_model_dir_or_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load
            full_sd = safe_load(pretrained_model_dir_or_checkpoint, device="cpu")
        else:
            full_sd = torch.load(pretrained_model_dir_or_checkpoint, mmap=True, weights_only=True, map_location="cpu")
        
        # Load directly into the un-wrapped model, matching the behavior in train_osp_RL_lora.py
        missing_keys, unexpected_keys = model.load_state_dict(full_sd, strict=False)
        if rank == 0:
            if missing_keys:
                print(f"[Base model checkpoint] missing_keys: {missing_keys[:20]}...")
            if unexpected_keys:
                print(f"[Base model checkpoint] unexpected_keys: {unexpected_keys[:20]}...")
        del full_sd
        has_loaded_pretrained_model = True

    if lora_path is not None:
        model = load_lora_and_merge(
            model=model,
            lora_path=lora_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            logger=logger,
            rank=rank,
        )

    # wrap model with fsdp2 mix-precision wrapper
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
        set_seed(seed, device_specific=False) # for init
        model.reset_parameters() # we should call reset_parameters because we init model at meta device 

    if explicit_prefetching_num_blocks > 0:
        set_modules_to_forward_prefetch(model.blocks, num_to_forward_prefetch=explicit_prefetching_num_blocks)

    log_on_main_process(logger, f"Diffusion model initialized, memory allocated: {get_memory_allocated()} GiB")

    pipeline = pipelines[pipeline_name](
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        predictor=model,
        scheduler=scheduler
    )

    prompts = load_prompts(prompt_txt)

    set_seed(seed, device_specific=True, process_group=dp_group)

    dp_rank = torch.distributed.get_rank(dp_group)
    dp_size = torch.distributed.get_world_size(dp_group)
    cp_rank = cp_state.global_cp_rank
    cp_size = cp_state.global_cp_size
    cp_group = cp_state.global_cp_group

    if len(prompts) % dp_size > 0:
        log_on_main_process(logger, f"Warning! Caused by using FSDP, we will pad some dummy data to make sure len(prompts) {len(prompts)} == dp_size {dp_size}.")
        while len(prompts) % dp_size > 0:
            prompts.append(prompts[0])

    video_grid = []
    for index in range(dp_rank * batch_size, len(prompts), batch_size * dp_size):
        batch_prompts = prompts[index: index + batch_size]
        videos = pipeline(
            prompt=batch_prompts,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            max_sequence_length=512,
            device=device
        )
        if cp_rank == 0:
            save_videos(videos, index, output_dir, save_fps)
            video_grid.append(videos)

    if len(video_grid) > 0:
        video_grid = torch.cat(video_grid, dim=0).to(device)

    if len(prompts) < batch_size * dp_size:
        active_ranks = range(len(prompts) // batch_size)
    else:
        active_ranks = range(dp_size)

    active_ranks = [x * cp_size for x in active_ranks]
    # torch.distributed.barrier()
    gathered_videos = gather_tensor_list_to_one([video_grid], group_dst=0, active_ranks=active_ranks)
    # torch.distributed.barrier()

    if rank == 0:
        video_grid = torch.cat(gathered_videos, dim=0)
        save_video_grid(video_grid, output_dir, fps=save_fps)
        print("Inference finished.")
        print(f"Saved {video_grid.shape[0]} samples to {output_dir}")

    cleanup_distributed_env()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/t2v.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
