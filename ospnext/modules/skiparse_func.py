# meta functions for skiparse attention
import torch
import torch.distributed as dist
from einops import rearrange, repeat as einops_repeat, reduce as einops_reduce
from ospnext.utils.utils import contiguous

def identity(x, grid_sizes=None, sparse_ratio=1):
    return x

def repeat(x, grid_sizes=None, sparse_ratio=1):
    x = einops_repeat(x, 'b n c -> (p b) n c', p=sparse_ratio)
    return x

def reduce(x, grid_sizes=None, sparse_ratio=1):
    x = einops_reduce(x, '(p b) n c -> b n c', 'mean', p=sparse_ratio)
    return x

def skiparse_2d_token(x, grid_sizes=None, sparse_ratio=1):
    T, H, W = grid_sizes
    return rearrange(x, 'b (t h p w q) c -> (p q b) (t h w) c', p=sparse_ratio, q=sparse_ratio, h=H // sparse_ratio, w=W // sparse_ratio)

def skiparse_2d_token_reverse(x, grid_sizes=None, sparse_ratio=1):
    T, H, W = grid_sizes
    return rearrange(x, '(p q b) (t h w) c -> b (t h p w q) c', p=sparse_ratio, q=sparse_ratio, h=H // sparse_ratio, w=W // sparse_ratio)

def skiparse_2d_group(x, grid_sizes=None, sparse_ratio=1):
    T, H, W = grid_sizes
    return rearrange(
        x, 'b (txh p1 p2 w q1 q2) c -> (p1 q1 b) (txh p2 w q2) c',
        p1=sparse_ratio, q1=sparse_ratio, p2=sparse_ratio, q2=sparse_ratio, w=W // (sparse_ratio ** 2)
    )

def skiparse_2d_group_reverse(x, grid_sizes=None, sparse_ratio=1):
    T, H, W = grid_sizes
    return rearrange(
        x, '(p1 q1 b) (txh p2 w q2) c -> b (txh p1 p2 w q1 q2) c',
        p1=sparse_ratio, q1=sparse_ratio, p2=sparse_ratio, q2=sparse_ratio, w=W // (sparse_ratio ** 2)
    )

# token2group is equivalent to group2token
def skiparse_2d_token_to_group(x, grid_sizes=None, sparse_ratio=1):
    T, H, W = grid_sizes
    # x: multiplication _: division
    return rearrange(
        x, '(p2 q2 b) (txh_p1 p1 w_q1 q1) c -> (p1 q1 b) (txh_p1 p2 w_q1 q2) c',
        p1=sparse_ratio, q1=sparse_ratio, p2=sparse_ratio, q2=sparse_ratio, w_q1=W // (sparse_ratio ** 2)
    )

def skiparse_2d_group_to_token(x, grid_sizes=None, sparse_ratio=1):
    # T, H, W = grid_sizes
    # return rearrange(
    #     x, '(p1 q1 b) (txh_p1 p2 w_q1 q2) c -> (p2 q2 b) (txh_p1 p1 w_q1 q1) c',
    #     p1=sparse_ratio, q1=sparse_ratio, p2=sparse_ratio, q2=sparse_ratio, h_p1=H // (sparse_ratio ** 2), w_q1=W // (sparse_ratio ** 2)
    # )
    return skiparse_2d_token_to_group(x, grid_sizes, sparse_ratio)


def _parallel_skiparse_2d_token_to_group(x, grid_sizes, sparse_ratio, group, group_size):
    T, H, W = grid_sizes
    P = sparse_ratio

    P2 = P * P
    G = P2 // group_size
    B_local = x.shape[0]  # = G * b
    b = B_local // G

    sub_grid_sizes = (T, H // P, W // P)

    # Step 1: local rearrange
    # [G*b, T*(H/P)*(W/P), C] → [P²*G*b, T*sub_H*sub_W, C]
    x = skiparse_2d_token(x, sub_grid_sizes, P)
    base, C = x.shape[1], x.shape[2]

    # Step 2: all_to_all
    x = contiguous(x).view(group_size, G, G * b, base, C)
    recv = torch.empty_like(x)
    dist.all_to_all_single(
        recv, x,
        group=group,
    )

    # Step 3: local permute
    # recv: [source, j_local, i_batch, b, base, C]
    # target: [source, i_local, j_local, b, base, C]
    recv = recv.view(group_size, G, G, b, base, C)
    recv = recv.permute(0, 2, 1, 3, 4, 5)
    x = contiguous(recv).view(P2 * G * b, base, C)

    # Step 4: local rearrange reverse
    # [P²*G*b, T*sub_H*sub_W, C] → [G*b, T*(H/P)*(W/P), C]
    x = skiparse_2d_token_reverse(x, sub_grid_sizes, P)

    return x

def _parallel_skiparse_2d_group_to_token(x, grid_sizes, sparse_ratio, group, group_size):
    return _parallel_skiparse_2d_token_to_group(x, grid_sizes, sparse_ratio, group, group_size)

class ParallelSkiparse2DToken2Group(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, grid_sizes, sparse_ratio, group):
        ctx.grid_sizes = grid_sizes
        ctx.sparse_ratio = sparse_ratio
        ctx.group = group
        ctx.size = size = dist.get_world_size(group)

        return _parallel_skiparse_2d_token_to_group(x, grid_sizes, sparse_ratio, group, size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _parallel_skiparse_2d_group_to_token(
            grad_output, ctx.grid_sizes, ctx.sparse_ratio, ctx.group, ctx.size
        )
        return grad_input, None, None, None

class ParallelSkiparse2DGroup2Token(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, grid_sizes, sparse_ratio, group):
        ctx.grid_sizes = grid_sizes
        ctx.sparse_ratio = sparse_ratio
        ctx.group = group
        ctx.size = size = dist.get_world_size(group)

        return _parallel_skiparse_2d_group_to_token(x, grid_sizes, sparse_ratio, group, size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _parallel_skiparse_2d_token_to_group(
            grad_output, ctx.grid_sizes, ctx.sparse_ratio, ctx.group, ctx.size
        )
        return grad_input, None, None, None

# Public function interfaces
def parallel_skiparse_2d_token_to_group(x, grid_sizes, sparse_ratio, group):
    return ParallelSkiparse2DToken2Group.apply(x, grid_sizes, sparse_ratio, group)

def parallel_skiparse_2d_group_to_token(x, grid_sizes, sparse_ratio, group):
    return ParallelSkiparse2DGroup2Token.apply(x, grid_sizes, sparse_ratio, group)