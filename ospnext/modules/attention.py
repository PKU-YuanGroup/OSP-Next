# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from ospnext.utils.utils import is_npu_available
from einops import rearrange

try:
    import flash_attn_interface
    from flash_attn_interface import (
        flash_attn_varlen_func as flash_attn_varlen_func_v3,
        flash_attn_func as flash_attn_func_v3
    )
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
    from flash_attn import (
        flash_attn_varlen_qkvpacked_func,
        flash_attn_varlen_func, 
        flash_attn_qkvpacked_func, 
        flash_attn_func
    )
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

from ospnext.utils.padding import pad_input, unpad_input

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'attention_with_mask',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2)
        return out

# adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
def flash_attn_no_pad(
    q,
    k,
    v,
    attn_mask=None,
    attn_mask_kv=None,
    is_cross_attn=False,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,
    deterministic=False,
):
    batch_size, seq_len_q, num_heads, _ = q.shape
    _, seq_len_kv, num_heads_kv, _ = k.shape

    if is_cross_attn:
        mask_q = attn_mask
        mask_kv = attn_mask_kv
    else:
        mask_q = attn_mask
        mask_kv = attn_mask

    no_mask_q = mask_q is None
    no_mask_kv = mask_kv is None

    # ==================================================================
    # Case 1: 完全无 mask
    # ==================================================================
    if (is_cross_attn and no_mask_kv) or (no_mask_q and no_mask_kv):
        if not is_cross_attn: # self-attn
            qkv = torch.stack([q, k, v], dim=2)
            return flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
        else:
            return flash_attn_func(
                q, k, v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )

    # ==================================================================
    # Case 2: 有 mask → 只 unpad 有 mask 的一侧
    # ==================================================================
    # --- 特殊路径: self-attn + 有 mask → qkvpacked ---
    if not is_cross_attn and not no_mask_q:
        qkv = torch.cat([q, k, v], dim=2)
        x = rearrange(qkv, "b s three_h d -> b s (three_h d)")
        x_unpad, indices, cu_seqlens, max_s = unpad_input(x, mask_q)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d",
            three=3, h=num_heads,
        )
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_seqlens, max_s, dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                indices, batch_size, seq_len_q,
            ),
            "b s (h d) -> b s h d", h=num_heads,
        )
        return output

    # --- 通用路径: cross-attn 或 mask 不同 ---
    # Q 侧
    if no_mask_q:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        indices_q = None
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seq_len_q, seq_len_q,
            device=q.device, dtype=torch.int32,
        )
        max_seqlen_q = seq_len_q
    else:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
            rearrange(q, "b s h d -> b s (h d)"), mask_q
        )
        q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=num_heads)

    # KV 侧
    if no_mask_kv:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_kv = torch.arange(
            0, (batch_size + 1) * seq_len_kv, seq_len_kv,
            device=k.device, dtype=torch.int32,
        )
        max_seqlen_kv = seq_len_kv
    else:
        k_unpad, _, cu_seqlens_kv, max_seqlen_kv = unpad_input(
            rearrange(k, "b s h d -> b s (h d)"), mask_kv
        )
        v_unpad, _, _, _ = unpad_input(
            rearrange(v, "b s h d -> b s (h d)"), mask_kv
        )
        k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=num_heads_kv)
        v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=num_heads_kv)

    output_unpad = flash_attn_varlen_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_kv,
        max_seqlen_q, max_seqlen_kv, dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )

    if indices_q is not None:
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                indices_q, batch_size, seq_len_q,
            ),
            "b s (h d) -> b s h d", h=num_heads,
        )
    else:
        output = rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    return output

def flash_attn_no_pad_v3(
    q,
    k,
    v,
    attn_mask=None,
    attn_mask_kv=None,
    is_cross_attn=False,
    causal=False,
    softmax_scale=None,
    deterministic=False,
):
    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seq_len_q, num_heads, _ = q.shape
    _, seq_len_kv, num_heads_kv, _ = k.shape

    if is_cross_attn:
        mask_q = attn_mask
        mask_kv = attn_mask_kv
    else:
        mask_q = attn_mask
        mask_kv = attn_mask

    no_mask_q = mask_q is None
    no_mask_kv = mask_kv is None

    # ==================================================================
    # Case 1: 完全无 mask → 不需要 unpad
    # 特别地，如果是 cross-attn，并且 KV 侧无 mask，则直接走 FA3 定长算子，避免 unpad 和 pad 的额外开销
    # ==================================================================
    if (is_cross_attn and no_mask_kv) or (no_mask_q and no_mask_kv):
        output, _ = flash_attn_func_v3(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        return output

    # ==================================================================
    # Case 2: 有 mask → 按需 unpad（只 unpad 有 mask 的一侧）
    # ==================================================================

    # --- self-attn + 有mask -> qkv一起cat再拆分，避免重复unpad ---
    if not is_cross_attn:
        qkv = torch.cat([q, k, v], dim=2)
        x = rearrange(qkv, "b s three_h d -> b s (three_h d)")
        x_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(x, mask_q)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d",
            three=3, h=num_heads,
        )
        q_unpad, k_unpad, v_unpad = x_unpad.unbind(dim=1)
        output_unpad, _ = flash_attn_varlen_func_v3(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_q,
            max_seqlen_q, max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
    else:
        # ---- 处理 Q 侧 ----
        if no_mask_q:
            q_unpad = rearrange(q, "b s h d -> (b s) h d")
            indices_q = None
            cu_seqlens_q = torch.arange(
                0, (batch_size + 1) * seq_len_q, seq_len_q,
                device=q.device, dtype=torch.int32,
            )
            max_seqlen_q = seq_len_q
        else:
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(
                rearrange(q, "b s h d -> b s (h d)"), mask_q
            )
            q_unpad = rearrange(q_unpad, "nnz (h d) -> nnz h d", h=num_heads)

        # ---- 处理 KV 侧 ----
        if no_mask_kv:
            k_unpad = rearrange(k, "b s h d -> (b s) h d")
            v_unpad = rearrange(v, "b s h d -> (b s) h d")
            cu_seqlens_kv = torch.arange(
                0, (batch_size + 1) * seq_len_kv, seq_len_kv,
                device=k.device, dtype=torch.int32,
            )
            max_seqlen_kv = seq_len_kv
        else:
            k_unpad, _, cu_seqlens_kv, max_seqlen_kv = unpad_input(
                rearrange(k, "b s h d -> b s (h d)"), mask_kv
            )
            v_unpad, _, _, _ = unpad_input(
                rearrange(v, "b s h d -> b s (h d)"), mask_kv
            )
            k_unpad = rearrange(k_unpad, "nnz (h d) -> nnz h d", h=num_heads_kv)
            v_unpad = rearrange(v_unpad, "nnz (h d) -> nnz h d", h=num_heads_kv)

        # ---- Flash Attention V3 计算 ----
        output_unpad, _ = flash_attn_varlen_func_v3(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q, cu_seqlens_kv,
            max_seqlen_q, max_seqlen_kv,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )

    # ---- 恢复 output 形状 ----
    if indices_q is not None:
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"),
                indices_q, batch_size, seq_len_q,
            ),
            "b s (h d) -> b s h d", h=num_heads,
        )
    else:
        output = rearrange(output_unpad, "(b s) h d -> b s h d", b=batch_size)

    return output

def scaled_dot_product_attention_with_mask(
    q, # [B, N, H, D]
    k, # [B, N, H, D]
    v, # [B, N, H, D]
    attn_mask=None,
    causal=False,
    dropout_p=0.,
):
    q = q.transpose(1, 2) # [B, N, H, D] -> [B, H, N, D]
    k = k.transpose(1, 2) # [B, N, H, D] -> [B, H, N, D]
    v = v.transpose(1, 2) # [B, N, H, D] -> [B, H, N, D]

    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        # [B, N] -> [B, 1, 1, N]，作为 key 侧 mask
        attn_mask = attn_mask[:, None, None, :]
        # NPU需要接受 [B, 1, Nq, Nkv] 形状的mask
        if is_npu_available():
            q_len = q.shape[2]
            attn_mask = attn_mask.expand(-1, -1, q_len, -1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

    return out.transpose(1, 2) # [B, H, N, D] -> [B, N, H, D]

def attention_with_mask(
    q, # [B, N, H, D]
    k, # [B, N, H, D]
    v, # [B, N, H, D]
    attn_mask=None,
    attn_mask_kv=None,
    is_cross_attn=False,
    causal=False,
    softmax_scale=None,
    deterministic=False,
    dropout_p=0.,
    dtype=torch.bfloat16,
):
    out_dtype = q.dtype
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    # ========== npu不支持flash attn接口，走SDPA ==========
    if is_npu_available() or (not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE):
        output = scaled_dot_product_attention_with_mask(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask_kv,
            causal=causal,
            dropout_p=dropout_p,
        )
    # ========== 优先用 Flash Attention V3 ==========
    elif FLASH_ATTN_3_AVAILABLE:
        output = flash_attn_no_pad_v3(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            attn_mask_kv=attn_mask_kv,
            is_cross_attn=is_cross_attn,
            causal=causal,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
        )
    # ========== 降级用 Flash Attention V2 ==========
    elif FLASH_ATTN_2_AVAILABLE:
        output = flash_attn_no_pad(
            q=q,
            k=k,
            v=v,
            attn_mask=attn_mask,
            attn_mask_kv=attn_mask_kv,
            is_cross_attn=is_cross_attn,
            causal=causal,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
        )
    else:
        raise ValueError(f"No supported attention backend found, FLASH_ATTN_2_AVAILABLE: {FLASH_ATTN_2_AVAILABLE}, FLASH_ATTN_3_AVAILABLE: {FLASH_ATTN_3_AVAILABLE}")

    return output.to(out_dtype)