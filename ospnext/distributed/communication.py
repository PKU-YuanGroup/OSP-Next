# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple, List, Optional
import torch
import torch.distributed as dist

from .cp_state import cp_state
from torch.nn import functional as F
from ospnext.utils.utils import contiguous


def get_shard_seq_lens(input: torch.Tensor, group: dist.ProcessGroup):
    seq_world_size = dist.get_world_size(group)
    local_len = torch.tensor(input.shape[1], dtype=torch.int32, device=input.device)
    all_lens = torch.empty(seq_world_size, dtype=torch.int32, device=input.device)
    dist.all_gather_into_tensor(all_lens, local_len, group=group)
    return all_lens.tolist()

def _all_to_all_4D(
    input: torch.Tensor, scatter_idx: int = 2, gather_idx: int = 1, group: dist.ProcessGroup = None,
    shard_seq_lens: List[int] = None,
) -> torch.Tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if seq_world_size == 1:
        return input

    if scatter_idx == 2 and gather_idx == 1:
        max_shard_seq_len = max(shard_seq_lens)
        local_shard_seq_len = shard_seq_lens[dist.get_rank(group)]
        local_gap = max_shard_seq_len - local_shard_seq_len
        if local_gap > 0:
            input = F.pad(input, (0, 0, 0, 0, 0, local_gap))

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = contiguous(
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        dist.all_to_all_single(output, input_t, group=group)
        # if scattering the seq-dim, transpose the heads back to the original dimension
        # output shape: [bs, seq_world_size, max_shard_seq_len, shard_hc, hs]
        output = output.transpose(0, 2).transpose(1, 2)
            
        has_variable_len = any(s != max_shard_seq_len for s in shard_seq_lens)
        if has_variable_len:
            chunks = [
                output[:, i].narrow(1, 0, shard_seq_lens[i])
                for i in range(seq_world_size)
            ]
            output = torch.cat(chunks, dim=1)  # [bs, total_valid_seq, shard_hc, hs]
        else:
            output = output.reshape(bs, -1, shard_hc, hs)
        return contiguous(output)

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        max_shard_seq_len = max(shard_seq_lens)
        has_variable_len = any(s != max_shard_seq_len for s in shard_seq_lens)
        if has_variable_len:
            chunks = input.split_with_sizes(shard_seq_lens, dim=1)
            padded_chunks = []
            for i, chunk in enumerate(chunks):
                gap = max_shard_seq_len - shard_seq_lens[i]
                if gap > 0:
                    padded_chunks.append(F.pad(chunk, (0, 0, 0, 0, 0, gap)))
                else:
                    padded_chunks.append(chunk)
            input_reshaped = torch.cat(padded_chunks, dim=1)
            shard_seqlen = max_shard_seq_len
        else:
            assert seqlen % seq_world_size == 0
            shard_seqlen = seqlen // seq_world_size
            input_reshaped = input

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = contiguous(
            input_reshaped.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        dist.all_to_all_single(output, input_t, group=group)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.reshape(hc, shard_seqlen, bs, hs).transpose(0, 2)

        local_shard_seq_len = shard_seq_lens[dist.get_rank(group)]
        if local_shard_seq_len < max_shard_seq_len:
            output = output[:, :local_shard_seq_len]

        return contiguous(output)

    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        shard_seq_lens: List[int],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.shard_seq_lens = shard_seq_lens

        return _all_to_all_4D(
            input, scatter_idx, gather_idx, group=group, shard_seq_lens=shard_seq_lens
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None, None]:
        grad_input = _all_to_all_4D(
            grad_output,
            scatter_idx=ctx.gather_idx,
            gather_idx=ctx.scatter_idx,
            group=ctx.group,
            shard_seq_lens=ctx.shard_seq_lens,
        )
        # 对应 forward 的 (group, input, scatter_idx, gather_idx, shard_seq_lens)
        return None, grad_input, None, None, None


def all_to_all_4D(
    input_: torch.Tensor, group: dist.ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1, shard_seq_lens: List[int] = None,
):
    return SeqAllToAll4D.apply(
        group, input_, scatter_dim, gather_dim, shard_seq_lens
    )


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support. """

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        input_size = list(input_.size())

        local_dim_size = torch.tensor(
            input_size[dim], dtype=torch.int32, device=input_.device
        )
        all_dim_sizes = torch.empty(
            world_size, dtype=torch.int32, device=input_.device
        )
        dist.all_gather_into_tensor(all_dim_sizes, local_dim_size, group=group)

        dim_sizes = all_dim_sizes.tolist()
        ctx.dim_sizes = dim_sizes

        all_same = all(s == dim_sizes[0] for s in dim_sizes)

        if all_same:
            tensor_list = [
                torch.empty(input_size, dtype=input_.dtype, device=input_.device) 
                for _ in range(world_size)
            ]
        else:
            sizes = [
                input_size[:dim] + [dim_sizes[i]] + input_size[dim + 1:]
                for i in range(world_size)
            ]
            tensor_list = [
                torch.empty(sizes[i], dtype=input_.dtype, device=input_.device)
                for i in range(world_size)
            ]
        dist.all_gather(tensor_list, contiguous(input_), group=group)
        return contiguous(torch.cat(tensor_list, dim=dim))

    @staticmethod
    def backward(ctx, grad_outputs):
        group = ctx.group
        dim = ctx.dim
        dim_sizes = ctx.dim_sizes
        rank = dist.get_rank(group)

        offset = sum(dim_sizes[:rank])
        grad_input = grad_outputs.narrow(dim, offset, dim_sizes[rank])

        return contiguous(grad_input), None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Wrapper with cleaner interface."""
    return _AllGather.apply(input_, dim, group)


class _AllToAllSingle(torch.autograd.Function):
    """All-to-all-single communication with autograd support.

    Forward:
        将 input_ 沿 dim 维按 input_split_sizes 切分, 第 i 块发给 rank i;
        从各 rank 收到的块按 output_split_sizes 拼接到 dim 维.

    Backward:
        对 grad_output 做一次 "逆向" all_to_all_single:
        将 grad_output 沿 dim 按 (前向的 output_split_sizes) 切分发回,
        按 (前向的 input_split_sizes) 接收, 还原为 grad_input.
    """

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        dim: int,
        group: Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        ctx.group = group
        ctx.dim = dim
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.input_shape = input_.shape

        if dim != 0:
            input_ = input_.transpose(0, dim)

        inp = contiguous(input_)
        if output_split_sizes is not None:
            out_size = list(inp.shape)
            out_size[0] = sum(output_split_sizes)
        else:
            out_size = list(inp.shape)          # 等分时输出与输入同形

        output = torch.empty(out_size, dtype=inp.dtype, device=inp.device)

        dist.all_to_all_single(
            output,
            inp,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        if dim != 0:
            output = output.transpose(0, dim)

        return contiguous(output)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        dim = ctx.dim

        bwd_output_split_sizes = ctx.input_split_sizes   # 反向的 recv sizes = 前向的 send sizes
        bwd_input_split_sizes = ctx.output_split_sizes    # 反向的 send sizes = 前向的 recv sizes

        if dim != 0:
            grad_output = grad_output.transpose(0, dim)

        grad_output = contiguous(grad_output)

        if bwd_output_split_sizes is not None:
            grad_input_size = list(grad_output.shape)
            grad_input_size[0] = sum(bwd_output_split_sizes)
        else:
            grad_input_size = list(grad_output.shape)

        grad_input = torch.empty(
            grad_input_size, dtype=grad_output.dtype, device=grad_output.device
        )

        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=bwd_output_split_sizes,
            input_split_sizes=bwd_input_split_sizes,
            group=group,
        )

        if dim != 0:
            grad_input = grad_input.transpose(0, dim)

        return contiguous(grad_input), None, None, None, None


def all_to_all_single(
    input_: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Differentiable wrapper around ``dist.all_to_all_single``. """

    return _AllToAllSingle.apply(
        input_, output_split_sizes, input_split_sizes, dim, group
    )
