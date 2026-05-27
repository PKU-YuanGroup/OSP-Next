# adapted form https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py 
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/padding.py

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(input, indices):
        assert input.ndim >= 2
        other_shape = input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0,
            repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, indices = inputs
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(values, indices, first_axis_dim):
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:],
            device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        values, indices, first_axis_dim = inputs
        ctx.save_for_backward(indices)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(input, indices):
        assert input.ndim >= 2
        output = input[indices]
        return output, input.detach()

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        input, indices = inputs
        ctx.save_for_backward(indices)
        ctx.first_axis_dim = input.shape[0]

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply



def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...)
        indices: (total_nnz)
        cu_seqlens: (batch + 1)
        max_seqlen_in_batch: int
    """

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    max_seqlen_in_batch = hidden_states.shape[1]

    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...)
        indices: (total_nnz)
        batch: int
        seqlen: int
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    hidden_states = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(hidden_states, "(b s) ... -> b s ...", b=batch)