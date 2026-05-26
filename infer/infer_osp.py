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
from ospnext.distributed.sp_state import sp_state
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
    # skiparse related
    sparse_ratio = model_config.get("sparse_ratio", 1)
    skiparse_model_type = model_config.get("skiparse_model_type", "full")
    is_skiparse_model = skiparse_model_type != "full"
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
    use_sequence_parallel = config.get("use_sequence_parallel", False)
    use_skiparse_sequence_parallel = config.get("use_skiparse_sequence_parallel", False)
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    explicit_prefetching_num_blocks = config.get("explicit_prefetching_num_blocks", 0)

    # save config
    output_dir = config.get("output_dir", "./output")
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
    # init sp mesh if use sequence parallel
    sp_size = 1
    use_sequence_parallel = use_sequence_parallel and config.get("sp_size", 1) > 1
    # skiparse sp
    skiparse_sp_size = 1
    use_skiparse_sequence_parallel = use_skiparse_sequence_parallel and config.get("skiparse_sp_size", 1) > 1 and sparse_ratio > 1
    use_global_sequence_parallel = use_sequence_parallel or use_skiparse_sequence_parallel
    global_sp_size = 1
    # full sp size
    full_sp_size = 1
    use_full_blocks_sequence_parallel = use_global_sequence_parallel and is_skiparse_model and num_full_blocks > 0
    if use_global_sequence_parallel:
        if use_sequence_parallel:
            sp_size = config.get("sp_size", 1)
        if use_skiparse_sequence_parallel:
            skiparse_sp_size = config.get("skiparse_sp_size", 1)
            if is_skiparse_model:
                # OSP-Next skiparse is 2D, so the per-rank shard must evenly divide sparse_ratio ** 2.
                assert skiparse_sp_size <= sparse_ratio ** 2 and (sparse_ratio ** 2) % skiparse_sp_size == 0
        global_sp_size = skiparse_sp_size * sp_size
        # dp * skiparse_sp * sp = world_size
        dp_global_sp_mesh = init_device_mesh("cuda", (world_size // global_sp_size, skiparse_sp_size, sp_size), mesh_dim_names=("dp", "skiparse_sp", "sp"))
        dp_group = dp_global_sp_mesh["dp"].get_group()
        global_sp_group = dp_global_sp_mesh["skiparse_sp", "sp"]._flatten().get_group()
        skiparse_sp_group = dp_global_sp_mesh["skiparse_sp"].get_group()
        # when initializing, full_blocks_sp_group and sp_group are the same group
        full_sp_group = sp_group = dp_global_sp_mesh["sp"].get_group()
        log_on_main_process(logger, f"We use sequence parallel, global_sp_size: {global_sp_size}, sp_size: {sp_size}, skiparse_sp_size: {skiparse_sp_size}")
        sp_state.reset(global_sp_group=global_sp_group, sp_group=sp_group, skiparse_sp_group=skiparse_sp_group, full_sp_group=full_sp_group)

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

    if use_sequence_parallel or use_full_blocks_sequence_parallel:
        if use_sequence_parallel and model.num_heads % sp_size != 0:
            raise ValueError(f"When using sequence parallel, num_heads {model.num_heads} mush be mutiple of sp_size {sp_size}!")
        if use_full_blocks_sequence_parallel:
            if global_sp_size <= model.num_heads and model.num_heads % global_sp_size == 0:
                full_sp_size = global_sp_size
            # find the greatest common divisor of model.num_heads and global_sp_size
            else:
                gcd = math.gcd(model.num_heads, global_sp_size)
                full_sp_size = gcd
            dummy_mesh = init_device_mesh("cuda", (world_size // full_sp_size, full_sp_size), mesh_dim_names=("dummy", "full_sp"))
            full_sp_group = dummy_mesh["full_sp"].get_group()
            sp_state.reset(full_sp_group=full_sp_group)
    
    model.eval()

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

    if not has_loaded_pretrained_model and pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        checkpointer = Checkpointer()
        checkpointer.load_model_from_path(model, pretrained_model_dir_or_checkpoint)
    else:
        raise ValueError(f"In inference mode, pretrained_model_dir_or_checkpoint {pretrained_model_dir_or_checkpoint} must be specified!")

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
    sp_rank = sp_state.global_sp_rank
    sp_size = sp_state.global_sp_size
    sp_group = sp_state.global_sp_group

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
        if sp_rank == 0:
            save_videos(videos, index, output_dir, save_fps)
            video_grid.append(videos)

    if len(video_grid) > 0:
        video_grid = torch.cat(video_grid, dim=0).to(device)

    if len(prompts) < batch_size * dp_size:
        active_ranks = range(len(prompts) // batch_size)
    else:
        active_ranks = range(dp_size)

    active_ranks = [x * sp_size for x in active_ranks]
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
