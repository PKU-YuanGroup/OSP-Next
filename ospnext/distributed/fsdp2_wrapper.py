import torch
from ospnext.utils.utils import is_npu_available, check_and_import_npu
check_and_import_npu()
import logging
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

def FSDP2_mix_wrapper(
    model,
    dp_mesh=None,
    weight_dtype=torch.bfloat16,
    main_block_to_half=None,
    blocks_to_float=None,
    blocks_to_output_float=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    is_rank_zero = torch.distributed.get_rank() == 0
    if is_rank_zero:
        logging.info("Parallelize Module with FSDP2...")
    low_precision_policy = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )
    high_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )
    fp32_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if blocks_to_output_float is not None and len(blocks_to_output_float) > 0:
        for module in model.modules():
            for block in blocks_to_output_float:
                if isinstance(module, block):
                    if is_rank_zero:
                        logging.info(f"FSDP {block} Module with All Float Precision to Output Float Results.")
                    fully_shard(module, mp_policy=fp32_precision_policy, **fsdp_kwargs)

    if blocks_to_float is not None and len(blocks_to_float) > 0:
        for module in model.modules():
            for block in blocks_to_float:
                if isinstance(module, block):
                    if is_rank_zero:
                        logging.info(f"FSDP {block} Module with High Precision.")
                    fully_shard(module, mp_policy=high_precision_policy, **fsdp_kwargs)

    if main_block_to_half is not None:
        for module in model.modules():
            if isinstance(module, main_block_to_half):
                if is_rank_zero:
                    logging.info(f"FSDP {main_block_to_half} Module with Low Precision.")
                fully_shard(module, mp_policy=low_precision_policy, **fsdp_kwargs)

    if is_rank_zero:
        logging.info(f"FSDP Other Modules.")
    fully_shard(model, mp_policy=low_precision_policy, **fsdp_kwargs)

    if is_rank_zero:
        logging.info("FSDP Down!")
        logging.info(f"Model Overview: \n{model}")


def FSDP2_fp32_wrapper(
    model,
    dp_mesh=None,
    main_block=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    fp32_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if main_block is not None:
        for module in model.modules():
            if isinstance(module, main_block):
                fully_shard(module, mp_policy=fp32_precision_policy, **fsdp_kwargs)
    fully_shard(model, mp_policy=fp32_precision_policy, **fsdp_kwargs)
