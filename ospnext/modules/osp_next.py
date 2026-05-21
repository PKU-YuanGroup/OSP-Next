# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from abc import ABC, abstractmethod
from einops import rearrange

from ospnext.distributed.cp_state import cp_state, use_context_parallel, use_skiparse_context_parallel, use_full_blocks_context_parallel
from ospnext.distributed.communication import all_gather, all_to_all_4D, all_to_all_single, get_shard_seq_lens
from ospnext.utils.utils import is_npu_available, safe_get_rank, contiguous, SafeCacheManager
from ospnext.utils.compile import maybe_compile

from .attention import flash_attention, attention, attention_with_mask
from .want2v import (
    sinusoidal_embedding_1d,
    WanLayerNorm as OSPNextLayerNorm,
    WanRMSNorm as OSPNextRMSNorm,
    Head,
)

from .skiparse_func import (
    identity,
    repeat,
    reduce,
    skiparse_2d_single,
    skiparse_2d_single_reverse,
    skiparse_2d_group,
    skiparse_2d_group_reverse,
    skiparse_2d_single_to_group,
    skiparse_2d_group_to_single,
    parallel_skiparse_2d_single_to_group,
    parallel_skiparse_2d_group_to_single,
)

from .hif8_linear import HIF8Linear
from .hif8_attention import hif8_attention_with_mask

def _make_linear(
    quant: str,
    in_features: int,
    out_features: int,
    scale_max_forward: float = 15.0,
    scale_max_backward: float = 224.0,
    bias: bool = True,
):
    if quant == "hif8":
        return HIF8Linear(
            in_features,
            out_features,
            bias=bias,
            scale_max_forward=scale_max_forward,
            scale_max_backward=scale_max_backward,
        )
    return nn.Linear(in_features, out_features, bias=bias)

T5_CONTEXT_TOKEN_NUMBER = 512

# 用实数rope，npu支持更好
@torch.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return freqs.float()  # (max_seq_len, dim//2)

class ContextParallelType:
    CP = "cp"
    SkiparseCP = "skiparse_cp"
    FullBlocksCP = "full_blocks_cp"
    GlobalCP = "global_cp"

class SkiparseModelType:
    DualEnd = "dual_end"
    Uniform = "uniform"
    Full = "full"

class SkiparseBlockType:
    Full = "full"
    Single = "single"
    Group = "group"

class RearrangeType:

    Identity = "identity"
    Repeat = "repeat"
    Reduce = "reduce"

    Skiparse2DSingle = "skiparse_2d_single"
    Skiparse2DSingleReverse = "skiparse_2d_single_reverse"
    Skiparse2DGroup = "skiparse_2d_group"
    Skiparse2DGroupReverse = "skiparse_2d_group_reverse"
    Skiparse2DSingle2Group = "skiparse_2d_single_to_group"
    Skiparse2DGroup2Single = "skiparse_2d_group_to_single"

    @classmethod
    def input_is_full(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingle,
            cls.Skiparse2DGroup,
        ]

    @classmethod
    def input_is_skiparse_2d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingleReverse,
            cls.Skiparse2DGroupReverse,
            cls.Skiparse2DSingle2Group,
            cls.Skiparse2DGroup2Single,
        ]
    
    @classmethod
    def output_is_full(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingleReverse,
            cls.Skiparse2DGroupReverse,
        ]


    @classmethod
    def output_is_skiparse_2d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingle,
            cls.Skiparse2DGroup,
            cls.Skiparse2DSingle2Group,
            cls.Skiparse2DGroup2Single,
        ]

    @classmethod
    def is_single2group(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingle2Group,
        ]
    
    @classmethod
    def is_group2single(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DGroup2Single,
        ]

# 包装纯函数为 nn.Module，便于torch.compile
class SkiparseRearrange(nn.Module):

    _SIMPLE_DISPATCH = {
        RearrangeType.Identity: identity,
        RearrangeType.Skiparse2DSingle: skiparse_2d_single,
        RearrangeType.Skiparse2DSingleReverse: skiparse_2d_single_reverse,
        RearrangeType.Skiparse2DGroup: skiparse_2d_group,
        RearrangeType.Skiparse2DGroupReverse: skiparse_2d_group_reverse,
        # 2d_single_to_group 和 2d_group_to_single 有分布式逻辑，单独处理
    }

    def __init__(self, sparse_ratio=1, rearrange_type=RearrangeType.Identity):
        super().__init__()

        self.sparse_ratio = sparse_ratio
        self.rearrange_type = rearrange_type

        self.skiparse_2d = "skiparse_2d" in self.rearrange_type

        if rearrange_type not in (
            RearrangeType.Identity, RearrangeType.Repeat, RearrangeType.Reduce
        ) and rearrange_type not in self._SIMPLE_DISPATCH and rearrange_type not in (
            RearrangeType.Skiparse2DSingle2Group, RearrangeType.Skiparse2DGroup2Single
        ):
            raise ValueError(f"Unsupported rearrange operation: {rearrange_type}")

        self.padding_cache = SafeCacheManager(max_cache_size=32)

    # ----------------- skiparse context parallel -----------------
    @maybe_compile(disable=True)
    def _skiparse_cp_scatter(self, x, dim=0):
        if not use_skiparse_context_parallel():
            return x
        size = x.shape[dim]
        cp_size = cp_state.skiparse_cp_size
        assert size % cp_size == 0
        chunk_size = size // cp_size
        return x.narrow(dim, cp_state.skiparse_cp_rank * chunk_size, chunk_size)

    @maybe_compile(disable=True)
    def _skiparse_cp_gather(self, x, dim=0):
        if not use_skiparse_context_parallel():
            return x
        x = all_gather(x, dim=dim, group=cp_state.skiparse_cp_group)
        return x

    @maybe_compile(disable=True)
    def _parallel_skiparse_2d_single_to_group(self, x, grid_sizes, sparse_ratio):
        return parallel_skiparse_2d_single_to_group(x, grid_sizes, sparse_ratio, cp_state.skiparse_cp_group)

    @maybe_compile(disable=True)
    def _parallel_skiparse_2d_group_to_single(self, x, grid_sizes, sparse_ratio):
        return parallel_skiparse_2d_group_to_single(x, grid_sizes, sparse_ratio, cp_state.skiparse_cp_group)

    def _dispatch_rearrange(self, x, grid_sizes):
        rt = self.rearrange_type
        sr = self.sparse_ratio

        # Repeat / Reduce：带分布式 scatter/gather
        if rt == RearrangeType.Repeat:
            return self._skiparse_cp_scatter(repeat(x, grid_sizes, sr))

        if rt == RearrangeType.Reduce:
            return reduce(self._skiparse_cp_gather(x), grid_sizes, sr)

        # 2d single⇄group：有 parallel 快路径
        if rt in (RearrangeType.Skiparse2DSingle2Group, RearrangeType.Skiparse2DGroup2Single):
            plain_fn, parallel_fn = {
                RearrangeType.Skiparse2DSingle2Group: (skiparse_2d_single_to_group, self._parallel_skiparse_2d_single_to_group),
                RearrangeType.Skiparse2DGroup2Single: (skiparse_2d_group_to_single, self._parallel_skiparse_2d_group_to_single),
            }[rt]
            if use_skiparse_context_parallel():
                return parallel_fn(x, grid_sizes, sr)
            return plain_fn(x, grid_sizes, sr)
        
        return self._SIMPLE_DISPATCH[rt](x, grid_sizes, sr)

    @maybe_compile(disable=True)
    def get_num_padding_tokens(self, grid_sizes):

        key = (grid_sizes, self.sparse_ratio, self.rearrange_type)
        if self.padding_cache.is_exist(key):
            return self.padding_cache.get(key)

        paddings = (0, 0)

        if self.skiparse_2d:
            block_size = self.sparse_ratio ** 2
            num_padding_tokens_h = (block_size - grid_sizes[1] % block_size) % block_size
            num_padding_tokens_w = (block_size - grid_sizes[2] % block_size) % block_size
            paddings = (num_padding_tokens_h, num_padding_tokens_w)

        self.padding_cache.set(key, paddings)
        return paddings

    def forward(self, x, grid_sizes=None):
        """
        带padding的skiparse rearrange。
        """

        if x is None:
            return x

        if self.rearrange_type in [RearrangeType.Identity, RearrangeType.Repeat, RearrangeType.Reduce]:
            return self._dispatch_rearrange(x, grid_sizes)

        B, N, C = x.shape
        x = contiguous(x)
        if RearrangeType.input_is_full(self.rearrange_type) and RearrangeType.output_is_skiparse_2d(self.rearrange_type):
            assert grid_sizes is not None and len(grid_sizes) == 3, "grid_sizes should be a tuple of (T, H, W)"
            T, H, W = grid_sizes
            num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                x = x.view(B, T, H, W, C)
                padding = (0, 0, 0, num_padding_tokens_w, 0, num_padding_tokens_h)
                x = F.pad(x, padding, mode="constant", value=0).view(B, -1, C)
                grid_sizes = (T, H + num_padding_tokens_h, W + num_padding_tokens_w)
            x = self._dispatch_rearrange(x, grid_sizes)
            x = self._skiparse_cp_scatter(x)
        elif RearrangeType.input_is_skiparse_2d(self.rearrange_type) and RearrangeType.output_is_full(self.rearrange_type):
            x = self._skiparse_cp_gather(x)
            T, H, W = grid_sizes
            num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                H = H + num_padding_tokens_h
                W = W + num_padding_tokens_w
                grid_sizes = (T, H, W)
            x = self._dispatch_rearrange(x, grid_sizes)
            B = x.shape[0]
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                H_orig = H - num_padding_tokens_h
                W_orig = W - num_padding_tokens_w
                x = contiguous(x.view(B, T, H, W, C)[:, :, :H_orig, :W_orig]).view(B, -1, C)
        elif RearrangeType.is_single2group(self.rearrange_type) or RearrangeType.is_group2single(self.rearrange_type):
            if self.skiparse_2d:
                T, H, W = grid_sizes
                num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
                grid_sizes = (T, H + num_padding_tokens_h, W + num_padding_tokens_w)
            x = self._dispatch_rearrange(x, grid_sizes)
        return x


class MetaPreprocessor(ABC):
    def __init__(
        self,
        is_skiparse_2d_model=False,
        sparse_ratio=4,
    ):
        self.is_skiparse_2d_model = is_skiparse_2d_model
        self.sparse_ratio = sparse_ratio
        if  not self.is_skiparse_2d_model and self.sparse_ratio > 1:
            warnings.warn("When skiparse_2d = False, sparse ratio should be 1, we instead use full attention.")
            self.sparse_ratio = 1
    
    @abstractmethod
    def preprocess(self, x, grid_sizes, **kwargs):
        pass
            
class ContextParallelPreprocessor(MetaPreprocessor):
    """
    在开启context parallel时，正常的skiparse会失效，因为skiparse需要获取全序列信息。
    一个合理的思路是，按照某种既定规则划分sequence，使得在序列内做skiparse等价于在全序列上做skiparse
    """

    def __init__(
        self,
        is_skiparse_2d_model=False,
        sparse_ratio=4,
    ):
        super().__init__(is_skiparse_2d_model, sparse_ratio)
        self.shard_seq_lens_cache = SafeCacheManager(max_cache_size=2)

    def check_short_sequence(self, num_tokens_or_sub_sequences: int, cp_size: int):
        if (num_tokens_or_sub_sequences % cp_size != 0 and
                num_tokens_or_sub_sequences <= (num_tokens_or_sub_sequences // cp_size + 1) * (cp_size - 1)):
            raise ValueError(
                f"Token {num_tokens_or_sub_sequences} is too short to be divided into {cp_size} parts"
            )

    def _skiparse_1d_params(self, H, W, cp_size):
        sub_len = self.sparse_ratio ** 2                      # 每个子模式长度
        num_sub = math.ceil(H * W / sub_len)                  # 子模式个数
        self.check_short_sequence(num_sub, cp_size)
        seq_len_per_cp = math.ceil(num_sub / cp_size) * sub_len
        return sub_len, num_sub, seq_len_per_cp

    def _skiparse_2d_params(self, H, W, cp_size):
        cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
        cp_size_w = cp_size // cp_size_h
        assert cp_size_h * cp_size_w == cp_size, (
            f"Unsupported cp_size={cp_size} for skiparse 2d. "
            f"Expected cp_size_h * cp_size_w == cp_size, got {cp_size_h} * {cp_size_w}."
        )
        sub_h = self.sparse_ratio ** 2
        sub_w = self.sparse_ratio ** 2
        num_sub_h = math.ceil(H / sub_h)
        num_sub_w = math.ceil(W / sub_w)
        self.check_short_sequence(num_sub_h, cp_size_h)
        self.check_short_sequence(num_sub_w, cp_size_w)
        seq_h = math.ceil(num_sub_h / cp_size_h) * sub_h
        seq_w = math.ceil(num_sub_w / cp_size_w) * sub_w
        return sub_h, sub_w, num_sub_h, num_sub_w, seq_h, seq_w

    # ----------------- preprocess -----------------

    def preprocess(self, x, grid_sizes, cp_type=ContextParallelType.CP):
        cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(cp_type)
        
        need_process = (use_context_parallel() or use_full_blocks_context_parallel()) and cp_size > 1
        if not need_process:
            return x, grid_sizes

        if cp_type is None or cp_type == ContextParallelType.FullBlocksCP:
            # 按照普通cp的方式直接在seq维度切分
            self.check_short_sequence(x.shape[1], cp_size)
            return contiguous(torch.chunk(x, cp_size, dim=1)[cp_rank]), grid_sizes

        x = contiguous(x)
        B, N, C = x.shape
        T, H, W = grid_sizes
        sub_grid_sizes = grid_sizes

        # skiparse 2d
        if self.is_skiparse_2d_model:
            x = x.view(B, T, H, W, C)
            _, _, _, _, seq_h, seq_w = self._skiparse_2d_params(H, W, cp_size)

            # 尽可能接近正方形，如果无法达到正方形，则让cp_size_w更大（因为一般用横屏视频）
            cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
            cp_size_w = cp_size // cp_size_h
            index_h = cp_rank // cp_size_w
            index_w = cp_rank % cp_size_w
            start_h = index_h * seq_h
            assert start_h < H, "The start index should be less than the height"
            end_h = min(start_h + seq_h, H)
            start_w = index_w * seq_w
            assert start_w < W, "The start index should be less than the width"
            end_w = min(start_w + seq_w, W)
            x = x[:, :, start_h:end_h, start_w:end_w, :]  # [B, T, seq_h, seq_w, C]
            sub_grid_sizes = (T, end_h - start_h, end_w - start_w)
            return contiguous(x).view(B, -1, C), sub_grid_sizes

        # 普通 cp
        self.check_short_sequence(N, cp_size)
        return contiguous(torch.chunk(x, cp_size, dim=1)[cp_rank]), sub_grid_sizes

    # ----------------- postprocess -----------------

    def postprocess(self, x, grid_sizes, shard_seq_lens=None, cp_type=ContextParallelType.CP):
        cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(cp_type)
        
        need_process = (use_context_parallel() or use_full_blocks_context_parallel()) and cp_size > 1
        if not need_process:
            return x

        if cp_type is None or cp_type == ContextParallelType.FullBlocksCP:
            return all_gather(x, dim=1, group=cp_group)

        x = contiguous(x)
        T, H, W = grid_sizes

        # ========== skiparse 2d 逆 ==========
        if self.is_skiparse_2d_model:
            _, _, _, _, seq_h, seq_w = self._skiparse_2d_params(H, W, cp_size)
            cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
            cp_size_w = cp_size // cp_size_h

            B, _, C = x.shape
            x = all_gather(x, dim=1, group=cp_group)
            x_list = x.split_with_sizes(shard_seq_lens, dim=1)
            x_out_w = []
            x_out = []
            for r in range(cp_size):
                index_h = r // cp_size_w
                index_w = r % cp_size_w
                start_h = index_h * seq_h
                end_h = min(start_h + seq_h, H)
                start_w = index_w * seq_w
                end_w = min(start_w + seq_w, W)

                h_len = end_h - start_h
                w_len = end_w - start_w
                x_out_w.append(contiguous(x_list[r]).view(B, T, h_len, w_len, C))
                if index_w == cp_size_w - 1 and len(x_out_w) == cp_size_w:
                    x_out.append(torch.cat(x_out_w, dim=3))
                    x_out_w = []
            x_out = torch.cat(x_out, dim=2)
            return contiguous(x_out).view(B, -1, C)

        # ========== 普通 cp 逆 ==========
        return all_gather(x, dim=1, group=cp_group)

    def get_shard_seq_lens(self, shape, grid_sizes, device="cuda", cp_type=ContextParallelType.CP):
        cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(cp_type)
        
        need_shard = (use_context_parallel() or use_full_blocks_context_parallel()) and cp_size > 1
        if not need_shard:
            return [[shape[1]] for _ in range(3)]

        key = (shape, grid_sizes, cp_type)
        if self.shard_seq_lens_cache.is_exist(key):
            return self.shard_seq_lens_cache.get(key)
        
        _, N, _ = shape
        
        dummy = torch.ones((1, N, 1), dtype=torch.bool, device=device)
        dummy, sub_grid_sizes = self.preprocess(dummy, grid_sizes, cp_type=cp_type)

        if cp_type is None or cp_type == ContextParallelType.FullBlocksCP:
            full_blocks_shard_seq_lens = get_shard_seq_lens(dummy, cp_state.full_cp_group)
            self.shard_seq_lens_cache.set(key, (full_blocks_shard_seq_lens, None, None))
            return [full_blocks_shard_seq_lens for _ in range(3)]

        full_shard_seq_lens = get_shard_seq_lens(dummy, cp_state.cp_group)

        if self.is_skiparse_2d_model:
            single_rearrange_type = RearrangeType.Skiparse2DSingle
            group_rearrange_type = RearrangeType.Skiparse2DGroup
        else:
            single_rearrange_type = RearrangeType.Identity
            group_rearrange_type = RearrangeType.Identity
        single_rearrange = SkiparseRearrange(self.sparse_ratio, single_rearrange_type)
        group_rearrange = SkiparseRearrange(self.sparse_ratio, group_rearrange_type)
        single_dummy = single_rearrange(dummy, sub_grid_sizes)
        group_dummy = group_rearrange(dummy, sub_grid_sizes)
        single_shard_seq_lens = get_shard_seq_lens(single_dummy, cp_state.cp_group)
        group_shard_seq_lens = get_shard_seq_lens(group_dummy, cp_state.cp_group)

        self.shard_seq_lens_cache.set(key, (full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens))

        return full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens


class SkiparseMaskPreprocessor(MetaPreprocessor):
    """
    生成skiparse attention下的attention mask。
    """

    def __init__(
        self, 
        is_skiparse_2d_model=False,
        sparse_ratio=4,
    ):
        super().__init__(is_skiparse_2d_model, sparse_ratio)
        self.cache = SafeCacheManager()

        if self.is_skiparse_2d_model:
            self.single_rearrange_type = RearrangeType.Skiparse2DSingle
            self.group_rearrange_type = RearrangeType.Skiparse2DGroup
        else:
            self.single_rearrange_type = RearrangeType.Identity
            self.group_rearrange_type = RearrangeType.Identity
        self.single_rearrange = SkiparseRearrange(self.sparse_ratio, self.single_rearrange_type)
        self.group_rearrange = SkiparseRearrange(self.sparse_ratio, self.group_rearrange_type)

    def _normalize_mask(self, mask):
        """将全 True 的 mask 归一化为 None"""
        if mask is None:
            return None
        # mask: (batch_size, seq_len), dtype=bool
        if mask.all():
            return None
        return mask

    def preprocess(self, shape, grid_sizes, context_preprocessor=None, dtype=torch.bool, device="cuda"):
        if not self.is_skiparse_2d_model or self.sparse_ratio == 1:
            return (None, None, None, None)

        key = (shape, grid_sizes, dtype, device)
        if self.cache.is_exist(key):
            return self.cache.get(key)

        B, N, _ = shape
        mask = torch.ones((B, N, 1), dtype=dtype, device=device)
        sub_grid_sizes = grid_sizes

        # 不用考虑full_blocks_cp，因为full block固定用None mask
        if use_context_parallel() and context_preprocessor is not None:
            mask, sub_grid_sizes = context_preprocessor.preprocess(mask, grid_sizes, cp_type=ContextParallelType.CP)

        local_single_mask = self.single_rearrange(mask, sub_grid_sizes)
        local_group_mask = self.group_rearrange(mask, sub_grid_sizes)

        global_single_mask = local_single_mask
        global_group_mask = local_group_mask
        if use_context_parallel() and context_preprocessor is not None:
            stacked = torch.cat([local_single_mask, local_group_mask], dim=-1)  # [B, N, 2]
            stacked = all_gather(stacked, dim=1, group=cp_state.cp_group)
            global_single_mask, global_group_mask = stacked.split(1, dim=-1)

        local_single_mask = local_single_mask.squeeze(-1).bool()
        local_group_mask = local_group_mask.squeeze(-1).bool()
        global_single_mask = global_single_mask.squeeze(-1).bool()
        global_group_mask = global_group_mask.squeeze(-1).bool()

        local_single_mask = self._normalize_mask(local_single_mask)
        local_group_mask = self._normalize_mask(local_group_mask)
        global_single_mask = self._normalize_mask(global_single_mask)
        global_group_mask = self._normalize_mask(global_group_mask)

        local_single_mask = contiguous(local_single_mask)
        local_group_mask = contiguous(local_group_mask)
        global_single_mask = contiguous(global_single_mask)
        global_group_mask = contiguous(global_group_mask)

        rank = safe_get_rank()
        if rank < cp_state.global_cp_size:
            print(f"=" * 20 + f" SkiparseMaskPreprocessor Cache Miss Rank #{rank}" + "=" * 20)
            print(f"local_single_mask is None: {local_single_mask is None}")
            print(f"local_group_mask is None: {local_group_mask is None}")
            print(f"global_single_mask is None: {global_single_mask is None}")
            print(f"global_group_mask is None: {global_group_mask is None}")
            print(f"=" * 20 + f" SkiparseMaskPreprocessor Cache Miss Rank #{rank}" + "=" * 20)

        self.cache.set(key, (local_single_mask, local_group_mask, global_single_mask, global_group_mask))
        return local_single_mask, local_group_mask, global_single_mask, global_group_mask

class SkiparseRopeWrapper:
    def __init__(self, freqs, context_preprocessor=None):
        self.freqs = freqs                                  # (max_len, head_dim), real float32
        self.context_preprocessor = context_preprocessor
        self.cache = SafeCacheManager(max_cache_size=4)
        self.real_dtype = torch.float32

    @maybe_compile(disable=True)
    def prepare_freqs(self, x, grid_sizes, head_dim, skiparse_rerrange=None, cp_type=ContextParallelType.CP):
        T, H, W = grid_sizes
        rearrange_type = (
            skiparse_rerrange.rearrange_type
            if skiparse_rerrange is not None
            else RearrangeType.Identity
        )
        key = (T, H, W, rearrange_type, cp_type, x.device)

        if self.cache.is_exist(key):
            return self.cache.get(key)

        freqs = self.freqs.split(
            [head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3],
            dim=1,
        )

        freqs_i = torch.cat(
            [
                freqs[0][:T].view(T, 1, 1, -1).expand(T, H, W, -1),
                freqs[1][:H].view(1, H, 1, -1).expand(T, H, W, -1),
                freqs[2][:W].view(1, 1, W, -1).expand(T, H, W, -1),
            ],
            dim=-1,
        ).reshape(1, T * H * W, -1)   # (1, T*H*W, head_dim)，注意存的是角度 θ

        # ----- Context Parallel 预处理 -----
        sub_grid_sizes = grid_sizes
        if use_context_parallel() and self.context_preprocessor is not None:
            freqs_i, sub_grid_sizes = self.context_preprocessor.preprocess(
                freqs_i, grid_sizes, cp_type=cp_type
            )

        # ----- Skiparse rearrange -----
        if skiparse_rerrange is not None:
            freqs_i = skiparse_rerrange(freqs_i, sub_grid_sizes)

        # ----- Context Parallel all-gather -----
        if use_context_parallel() and self.context_preprocessor is not None:
            cp_group, _, _ = cp_state.get_cp_infos_with_type(cp_type)
            freqs_i = all_gather(freqs_i, dim=1, group=cp_group)

        freqs_i = freqs_i.to(self.real_dtype)
        cos_f = freqs_i.cos().unsqueeze(2)   # (1, seq, 1, head_dim)
        sin_f = freqs_i.sin().unsqueeze(2)   # (1, seq, 1, head_dim)

        result = (cos_f, sin_f)
        self.cache.set(key, result)
        return result

    # ----- 实数 RoPE -----
    @torch.autocast("cuda", enabled=False)
    def apply_rope(self, x, grid_sizes, skiparse_rerrange=None, cp_type=ContextParallelType.CP):
        """
        x: (B, seq_len, num_heads, head_dim * 2)
        """
        B, seq_len, num_heads, head_dim = x.shape
        head_dim = head_dim // 2

        x = x.to(self.real_dtype).reshape(-1, seq_len, num_heads, head_dim, 2)
        x1 = x[..., 0]   # (B, seq, heads, head_dim) # 实部
        x2 = x[..., 1]   # (B, seq, heads, head_dim) # 虚部

        # 获取 cos/sin
        cos_f, sin_f = self.prepare_freqs(
            x, grid_sizes, head_dim, skiparse_rerrange, cp_type
        )

        # cos_f/sin_f shape: (freqs_B, seq, 1, head_dim)
        # x1/x2 shape:       (B,       seq, heads, head_dim)
        # 当 skiparse rearrange 后 freqs_B = P²，x 的 B = P²*b
        # 需要把 freqs repeat b 次对齐
        freqs_B = cos_f.shape[0]
        if freqs_B > 1 and B != freqs_B:
            # B = P² * b, freqs_B = P²
            assert B % freqs_B == 0, f"B={B} must be divisible by freqs_B={freqs_B}"
            b = B // freqs_B
            # (P², seq, 1, head_dim) -> (P²*b, seq, 1, head_dim)
            cos_f = cos_f.repeat_interleave(b, dim=0)
            sin_f = sin_f.repeat_interleave(b, dim=0)

        # 复数乘法的实数等价：
        # (x1 + x2·i)(cosθ + sinθ·i) = (x1·cos - x2·sin) + (x1·sin + x2·cos)·i
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x1 * sin_f + x2 * cos_f

        # 交织回原始 layout
        out = torch.stack([o1, o2], dim=-1).flatten(3)  # (B, seq, heads, head_dim*2)
        return out.float()

class OSPNextSelfAttention(nn.Module):

    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=(-1, -1), 
        qk_norm=True, 
        eps=1e-6, 
        # skiparse相关
        sparse_ratio=2,
        skiparse_block_type=SkiparseBlockType.Full,
        # hif8 quantization
        quant=None,
        quant_attn=None,
        scale_max_forward=15.0,
        scale_max_backward=224.0,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.quant_attn = quant_attn
        self.scale_max_forward = scale_max_forward
        self.scale_max_backward = scale_max_backward

        # layers
        self.q = _make_linear(quant, dim, dim, scale_max_forward, scale_max_backward)
        self.k = _make_linear(quant, dim, dim, scale_max_forward, scale_max_backward)
        self.v = _make_linear(quant, dim, dim, scale_max_forward, scale_max_backward)
        self.o = _make_linear(quant, dim, dim, scale_max_forward, scale_max_backward)
        self.norm_q = OSPNextRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = OSPNextRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.skiparse_block_type = skiparse_block_type
        if self.skiparse_block_type == SkiparseBlockType.Full:
            self.rearrange_rope = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
        elif self.skiparse_block_type == SkiparseBlockType.Single:
            self.rearrange_rope = SkiparseRearrange(
                sparse_ratio=sparse_ratio,
                rearrange_type=RearrangeType.Skiparse2DSingle,
            )
        elif self.skiparse_block_type == SkiparseBlockType.Group:
            self.rearrange_rope = SkiparseRearrange(
                sparse_ratio=sparse_ratio,
                rearrange_type=RearrangeType.Skiparse2DGroup,
            )

        if self.skiparse_block_type == SkiparseBlockType.Full:
            self.cp_type = ContextParallelType.FullBlocksCP
        else:
            self.cp_type = ContextParallelType.CP
            

    @maybe_compile(disable=True)
    def pre_self_attn_all_to_all(
        self, 
        q, k, v,
        shard_seq_lens=None,
    ):

        if use_context_parallel() or use_full_blocks_context_parallel():
            cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(self.cp_type)
            if cp_size > 1:
                q = all_to_all_4D(q, group=cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
                k = all_to_all_4D(k, group=cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
                v = all_to_all_4D(v, group=cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
        return q, k, v


    @maybe_compile(disable=True)
    def post_self_attn_all_to_all(
        self,
        x,
        shard_seq_lens=None,
    ):
        if use_context_parallel() or use_full_blocks_context_parallel():
            cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(self.cp_type)
            if cp_size > 1:
                x = all_to_all_4D(x, group=cp_group, scatter_dim=1, gather_dim=2, shard_seq_lens=shard_seq_lens)
        return x

    def forward(
        self, 
        x, 
        attn_mask,
        grid_sizes_for_rope,
        rope_wrapper,
        shard_seq_lens=None,
    ):
        B, N, H, D = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(B, N, H, D)
        k = self.norm_k(self.k(x)).view(B, N, H, D)
        v = self.v(x).view(B, N, H, D)


        q, k, v = self.pre_self_attn_all_to_all(
            q, k, v, shard_seq_lens
        )

        q = rope_wrapper.apply_rope(q, grid_sizes_for_rope, self.rearrange_rope, cp_type=self.cp_type)
        k = rope_wrapper.apply_rope(k, grid_sizes_for_rope, self.rearrange_rope, cp_type=self.cp_type)

        if self.quant_attn == "hif8":
            x = hif8_attention_with_mask(
                q,
                k,
                v,
                attn_mask=attn_mask,
                attn_mask_kv=attn_mask,
                scale_max_forward=self.scale_max_forward,
                scale_max_backward=self.scale_max_backward,
            )
        else:
            x = attention_with_mask(
                q,
                k, 
                v, 
                # self attn，q/k/v共享mask
                attn_mask=attn_mask,
                attn_mask_kv=attn_mask,
            )

        x = self.post_self_attn_all_to_all(x, shard_seq_lens=shard_seq_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class OSPNextCrossAttention(OSPNextSelfAttention):

    def forward(
        self, 
        x, 
        attn_mask,
        text,
    ):

        B, N, H, D = *x.shape[:2], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(B, -1, H, D)
        k = self.norm_k(self.k(text)).view(B, -1, H, D)
        v = self.v(text).view(B, -1, H, D)

        # NOTE cross attn不需要all2all，因为img只作为q，天然支持按序列切分计算

        # cross attn，q作为query，k/v作为key/value，k/v没有mask
        if self.quant_attn == "hif8":
            x = hif8_attention_with_mask(
                q,
                k,
                v,
                attn_mask=attn_mask,
                attn_mask_kv=None,
                is_cross_attn=True,
                scale_max_forward=self.scale_max_forward,
                scale_max_backward=self.scale_max_backward,
            )
        else:
            x = attention_with_mask(q, k, v, attn_mask=attn_mask, attn_mask_kv=None, is_cross_attn=True)

        # NOTE cross attn不需要all2all，因为img只作为q，天然支持按序列切分计算
        
        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class OSPNextAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        # skiparse相关
        sparse_ratio=2,
        skiparse_block_type=SkiparseBlockType.Full,
        is_full2skiparse_block=False,
        is_skiparse2full_block=False,
        # hif8 quantization
        quant=None,
        quant_attn=None,
        scale_max_forward=15.0,
        scale_max_backward=224.0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = OSPNextLayerNorm(dim, eps)
        self.self_attn = OSPNextSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            sparse_ratio=sparse_ratio,
            skiparse_block_type=skiparse_block_type,
            quant=quant,
            quant_attn=quant_attn,
            scale_max_forward=scale_max_forward,
            scale_max_backward=scale_max_backward,
        )
        self.norm3 = (
            OSPNextLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = OSPNextCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps,
            quant=quant,
            quant_attn=quant_attn,
            scale_max_forward=scale_max_forward,
            scale_max_backward=scale_max_backward,
        )
        self.norm2 = OSPNextLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            _make_linear(quant, dim, ffn_dim, scale_max_forward, scale_max_backward),
            nn.GELU(approximate="tanh"),
            _make_linear(quant, ffn_dim, dim, scale_max_forward, scale_max_backward),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.sparse_ratio = sparse_ratio
        self.skiparse_block_type = skiparse_block_type
        self.is_skiparse2full_block = is_skiparse2full_block
        self.is_full2skiparse_block = is_full2skiparse_block

        if self.skiparse_block_type == SkiparseBlockType.Full:
            self.rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.context_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
        else:
            self.context_rearrange_input = SkiparseRearrange(
                sparse_ratio=self.sparse_ratio ** 2, 
                rearrange_type=RearrangeType.Repeat
            )
            if self.skiparse_block_type == SkiparseBlockType.Single:
                if self.is_full2skiparse_block:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse2DSingle,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                else:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse2DGroup2Single,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            elif self.skiparse_block_type == SkiparseBlockType.Group:
                if self.is_skiparse2full_block:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse2DSingle2Group,
                    )
                    self.rearrange_output = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse2DGroupReverse,
                    )
                else:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse2DSingle2Group,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)

    def block_forward(
        self,
        x,
        attn_mask,
        cross_attn_mask,
        e,
        grid_sizes_for_rope,
        rope_wrapper,
        text,
        shard_seq_lens=None,
    ):
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], 
            attn_mask, 
            grid_sizes_for_rope, 
            rope_wrapper, 
            shard_seq_lens=shard_seq_lens,
        )
        x = x + y * e[2]
        # cross-attention & ffn function
        x = x + self.cross_attn(
            self.norm3(x), 
            cross_attn_mask,
            text, 
        )
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x

    @maybe_compile()
    def forward(
        self,
        x,
        attn_mask,
        cross_attn_mask,
        e,
        sub_grid_sizes,
        grid_sizes_for_rope,
        rope_wrapper,
        text,
        shard_seq_lens=None,
        gradient_checkpointing=False,
    ):
        # 对输入做rerrange
        x = self.rearrange_input(x, grid_sizes=sub_grid_sizes)
        text = self.context_rearrange_input(text)
        e = self.context_rearrange_input(e)

        if gradient_checkpointing and torch.is_grad_enabled():
            # 收集本 block 内所有 LoRA 参数，显式传入 checkpoint，
            # 确保 checkpoint 的 autograd 能正确追踪 LoRA 参数的梯度。
            # 不这样做的话，use_reentrant=False 的 checkpoint 在 FSDP2 环境下
            # 无法感知 LoRA 参数参与了计算，导致 LoRA 梯度为零。
            _lora_params = [p for n, p in self.named_parameters() if 'lora_' in n]

            def _checkpointed_block_forward(x, attn_mask, cross_attn_mask, e,
                                            grid_sizes_for_rope, rope_wrapper, text,
                                            *_lora_dummy_args,
                                            shard_seq_lens=None):
                return self.block_forward(
                    x, attn_mask, cross_attn_mask, e,
                    grid_sizes_for_rope, rope_wrapper, text,
                    shard_seq_lens=shard_seq_lens,
                )

            x = torch.utils.checkpoint.checkpoint(
                _checkpointed_block_forward,
                x,
                attn_mask,
                cross_attn_mask,
                e,
                grid_sizes_for_rope,
                rope_wrapper,
                text,
                *_lora_params,
                shard_seq_lens=shard_seq_lens,
                use_reentrant=False,
            )
        else:
            x = self.block_forward(
                x,
                attn_mask,
                cross_attn_mask,
                e,
                grid_sizes_for_rope,
                rope_wrapper,
                text,
                shard_seq_lens=shard_seq_lens,
            )

        # 对输出做rerrange
        x = self.rearrange_output(x, grid_sizes=sub_grid_sizes)

        return x


class OSPNextModel(ModelMixin, ConfigMixin):

    r"""
    OSPNext model. 基于WanT2V，添加了skiparse机制。
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # skiparse相关参数
        skiparse_model_type=SkiparseModelType.Full,
        sparse_ratio=1,
        num_full_blocks=0,
        # hif8 quantization (None or "hif8")
        quant=None,
        quant_attn=None,
        scale_max_forward=15.0,
        scale_max_backward=224.0,
        **kwargs,
    ):

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.gradient_checkpointing = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.skiparse_model_type = skiparse_model_type
        self.sparse_ratio = sparse_ratio
        self.num_full_blocks = num_full_blocks

        if self.skiparse_model_type == SkiparseModelType.Full:
            self.sparse_ratio = 1
            self.num_full_blocks = self.num_layers
            full_block_indices = list(range(0, self.num_layers))
        else:
            assert self.num_layers % 2 == 0 and self.num_full_blocks % 2 == 0 and self.num_full_blocks <= self.num_layers // 2, "num_full_blocks should be divisible by 2 and less than or equal to num_layers // 2"
            if self.num_full_blocks == 0:
                full_block_indices = []
            elif self.skiparse_model_type == SkiparseModelType.DualEnd:
                assert self.num_full_blocks % 4 == 0, "num_full_blocks should be divisible by 4"
                skiparse_start_index = self.num_full_blocks // 2
                skiparse_end_index = self.num_layers - self.num_full_blocks // 2 - 1
                assert skiparse_start_index < skiparse_end_index, "skiparse_start_index should be less than skiparse_end_index"
                full_block_indices = list(range(0, skiparse_start_index)) + list(range(skiparse_end_index + 1, self.num_layers))
            else:
                raise ValueError(f"Unsupported skiparse model type: {self.skiparse_model_type}")

        # 添加虚拟的两端full block，方便处理边界情况
        full_block_indices = [-1] + full_block_indices + [self.num_layers]

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i in full_block_indices:
                skiparse_block_type = SkiparseBlockType.Full
            elif i % 2 == 0:
                skiparse_block_type = SkiparseBlockType.Single
            else:
                skiparse_block_type = SkiparseBlockType.Group
            # 当前block是sparse，上一个block是full
            is_full2skiparse_block = (i - 1) in full_block_indices and i not in full_block_indices
            # 当前block是sparse，下一个block是full
            is_skiparse2full_block = (i + 1) in full_block_indices and i not in full_block_indices
            self.blocks.append(
                OSPNextAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    sparse_ratio=sparse_ratio if skiparse_block_type is not SkiparseBlockType.Full else 1,
                    skiparse_block_type=skiparse_block_type,
                    is_full2skiparse_block=is_full2skiparse_block,
                    is_skiparse2full_block=is_skiparse2full_block,
                    quant=quant,
                    quant_attn=quant_attn,
                    scale_max_forward=scale_max_forward,
                    scale_max_backward=scale_max_backward,
                )
            )

        # 初始化block结束后删除两个dummy full block
        self.full_block_indices = full_block_indices[1:-1]

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.rope_d = dim // num_heads
        self.freqs = None
        self.rope_wrapper = None


        self.mask_preprocessor = SkiparseMaskPreprocessor(
            is_skiparse_2d_model=self.skiparse_model_type != SkiparseModelType.Full,
            sparse_ratio=self.sparse_ratio,
        )

        # cp管理器。要兼容skiparse attn的情况下，Ulysses cp需要按照特定规则切分序列
        self.context_preprocessor = ContextParallelPreprocessor(
            is_skiparse_2d_model=self.skiparse_model_type != SkiparseModelType.Full,
            sparse_ratio=self.sparse_ratio,
        )

        # full block部分无法采用SSP，所以采用Ulysses SP
        self.need_full_blocks_context_parallel = self.skiparse_model_type != SkiparseModelType.Full and self.num_full_blocks > 0

        """
        skiparse cp 的逻辑完全写在了skiparse rearrange中，会根据当前block type自动触发
        all_to_all, gather, scatter。
        因此我们只需要关注ulysses cp下该怎么操作序列
        """
        # main_cp_type指大部分sparse block采用的self attn处的cp
        if self.skiparse_model_type == SkiparseModelType.Full:
            self.main_cp_type = ContextParallelType.FullBlocksCP
        else:
            self.main_cp_type = ContextParallelType.CP

        # final_cp_type指最后一个block采用的self attn处的cp，用于还原序列
        if self.blocks[-1].skiparse_block_type == SkiparseBlockType.Full:
            self.final_cp_type = ContextParallelType.FullBlocksCP
        else:
            self.final_cp_type = ContextParallelType.CP

        if safe_get_rank() == 0:
            print(f"=" * 20 + f"OSPNextModel init" + "=" * 20)
            print(f"skiparse_model_type: {self.skiparse_model_type}")
            print(f"sparse_ratio: {self.sparse_ratio}")
            print(f"num_full_blocks: {self.num_full_blocks}")
            print(f"need_full_blocks_context_parallel: {self.need_full_blocks_context_parallel}")
            print(f"full_block_indices: {self.full_block_indices}")
            print(f"=" * 20 + f"OSPNextModel init" + "=" * 20)

        # initialize weights
        self.init_weights()

    def set_gradient_checkpointing(self, enabled = False):
        self.gradient_checkpointing = enabled 

    def reset_parameters(self):
        print(f"{__class__.__name__} reset parameters!")
        self.init_weights()

    def forward(
        self,
        x, # [B C T H W]
        t, # [B]
        text, # [B N C]
        **kwargs,
    ):

        # params
        device = self.patch_embedding.weight.device
        use_full_blocks_cp = use_full_blocks_context_parallel() and self.need_full_blocks_context_parallel

        # maybe we use meta device for init, so rope freqs should init before forward
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        if self.freqs is None:
            self.freqs = torch.cat(
                [
                    rope_params(1024, self.rope_d - 4 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                ],
                dim=1,
            ).to(device)
            self.rope_wrapper = SkiparseRopeWrapper(self.freqs, self.context_preprocessor)

        # embeddings
        x = self.patch_embedding(x)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        x, grid_sizes = self.patchify(x)
        grid_sizes_for_rope = grid_sizes
        patchify_x_shape = x.shape

        # 计算shard_seq_lens，在all2all中用于恢复原序列长度
        full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens = self.context_preprocessor.get_shard_seq_lens(
            patchify_x_shape, grid_sizes, device=device, cp_type=self.main_cp_type
        )
        full_block_full_shard_seq_lens = full_shard_seq_lens
        if use_full_blocks_cp:
            full_block_full_shard_seq_lens, _, _ = self.context_preprocessor.get_shard_seq_lens(
                patchify_x_shape, grid_sizes, device=device, cp_type=ContextParallelType.FullBlocksCP
            )

        # skiparse过程中可能有padding，所以需要生成mask
        # mask需要传入context preprocessor，以对齐cp切分后的seq
        local_single_mask, local_group_mask, global_single_mask, global_group_mask = self.mask_preprocessor.preprocess(
            patchify_x_shape, grid_sizes, context_preprocessor=self.context_preprocessor,
            dtype=torch.bool, device=device
        )
        
        x, sub_grid_sizes = self.context_preprocessor.preprocess(
            x, grid_sizes,
            cp_type=self.main_cp_type if not use_full_blocks_cp else ContextParallelType.FullBlocksCP
        )

        # text
        text = self.text_embedding(text)

        # 最终还原postprocess的shard_seq_lens和cp_type
        if self.final_cp_type == ContextParallelType.CP:
            final_shard_seq_lens = full_shard_seq_lens
        elif self.final_cp_type == ContextParallelType.FullBlocksCP:
            final_shard_seq_lens = full_block_full_shard_seq_lens
        for idx, block in enumerate(self.blocks):
            # 如果是full2skiparse block，且采用了full blocks cp
            # 需要先在full blocks cp group中得到完整序列，再重新在cp group中shard
            if use_full_blocks_cp:
                # 不是第一个block，且从full切换到sparse，则重新切分
                if idx != 0 and block.is_full2skiparse_block:
                    x = self.context_preprocessor.postprocess(
                        x, grid_sizes, shard_seq_lens=full_block_full_shard_seq_lens, cp_type=ContextParallelType.FullBlocksCP
                    )
                    x, sub_grid_sizes = self.context_preprocessor.preprocess(
                        x, grid_sizes, cp_type=self.main_cp_type
                    )

            if block.skiparse_block_type == SkiparseBlockType.Full:
                attn_mask, cross_attn_mask = None, None
                shard_seq_lens = full_shard_seq_lens if not use_full_blocks_cp else full_block_full_shard_seq_lens
            elif block.skiparse_block_type == SkiparseBlockType.Single:
                attn_mask, cross_attn_mask, shard_seq_lens = global_single_mask, local_single_mask, single_shard_seq_lens
            elif block.skiparse_block_type == SkiparseBlockType.Group:
                attn_mask, cross_attn_mask, shard_seq_lens = global_group_mask, local_group_mask, group_shard_seq_lens
            
            x = block(
                x, 
                attn_mask, 
                cross_attn_mask,
                e0, 
                sub_grid_sizes, 
                grid_sizes_for_rope, 
                self.rope_wrapper, 
                text,
                shard_seq_lens, 
                gradient_checkpointing=self.gradient_checkpointing,
            )

            # 如果是skiparse2full block，且采用了full blocks cp
            # 需要先在cp group中得到完整序列，再重新在full blocks cp group中shard
            if use_full_blocks_cp:
                # 不是最后一个block，后续还有full blocks，则重新切分
                if idx != len(self.blocks) - 1 and block.is_skiparse2full_block:
                    x = self.context_preprocessor.postprocess(
                        x, grid_sizes, shard_seq_lens=full_shard_seq_lens, cp_type=self.main_cp_type
                    )
                    x, sub_grid_sizes = self.context_preprocessor.preprocess(
                        x, grid_sizes, cp_type=ContextParallelType.FullBlocksCP
                    )

        # head
        x = self.head(x, e)

        # 根据final_shard_seq_lens和final_cp_type还原序列
        x = self.context_preprocessor.postprocess(
            x, grid_sizes, 
            shard_seq_lens=final_shard_seq_lens, 
            cp_type=self.final_cp_type
        )

        # unpatchify
        x = self.unpatchify(x, *grid_sizes)
        return x.float()

    def patchify(self, embs):
        # get f, h, w from b c f h w
        grid_sizes = embs.shape[2:]

        # b c f h w  -> b (f h w) c
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c")

        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        # b (f h w) (x y z c) -> b c (f x) (h y) (w z)
        patch_out = rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames,
            h=height,
            w=width,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return patch_out

    def init_weights(self):
        for n, m in self.named_modules():
            if n == "":
                continue
            if hasattr(m, "reset_parameters"):
                # print(f"{n} -> reset_parameters")
                m.reset_parameters()
        

models = {
    "osp_next": OSPNextModel
}

models_main_block = {
    "osp_next": OSPNextAttentionBlock
}

models_blocks_to_float = {
    "osp_next": [OSPNextLayerNorm, OSPNextRMSNorm]
}

models_blocks_to_output_float = {
    "osp_next": None
}


if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    model = OSPNextModel().to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    x = torch.randn(2, 16, 21, 60, 104, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 512, 4096, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x, t, context)
    print(y.shape)