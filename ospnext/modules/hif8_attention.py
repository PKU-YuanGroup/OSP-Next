# HIF8 quantization-aware attention for DiT (OSP-Next, NPU path)
#
# Applies HIF8 quant-dequant simulation to Q, K, V tensors BEFORE the
# core attention kernel (SDPA / FA), and quantizes the incoming gradients
# (dL/dQ, dL/dK, dL/dV) in the backward pass.
#
# This sits one level above hif8_linear.py:
#   - hif8_linear.py  : quantizes activation × weight in the projection layers
#   - hif8_attention.py: quantizes Q/K/V entering the softmax(QKᵀ/√d)·V kernel
#
# Call graph inside OSPNextSelfAttention.forward():
#
#   x  ──► self.q/k/v  (HIF8Linear, or nn.Linear)
#        ──► Q, K, V  [B, N, H, D]
#        ──► hif8_attention_with_mask(Q, K, V, ...)   ◄── this file
#                ├─ _HIF8QKVQuantFunction (quantize Q/K/V forward; quantize grad backward)
#                └─ attention_with_mask(Q_q, K_q, V_q, ...)  ◄── attention.py (SDPA / FA)
#
# Forward quantization:
#   Q_q = quant_dequant_hif8(Q * scale_q) / scale_q,  scale_q = scale_max_forward / max|Q|
#   K_q = quant_dequant_hif8(K * scale_k) / scale_k,  scale_k = scale_max_forward / max|K|
#   V_q = quant_dequant_hif8(V * scale_v) / scale_v,  scale_v = scale_max_forward / max|V|
#   out = attention(Q_q, K_q, V_q)
#
# Backward quantization (gradients w.r.t. Q/K/V from attention backward):
#   dQ_q = quant_dequant_hif8(dQ * sg) / sg,  sg = scale_max_backward / max|dQ|
#   dK_q, dV_q – same treatment

import torch
from typing import Optional

# Re-use the per-tensor quant helper from hif8_linear to avoid duplication.
from .hif8_linear import _quant
from .attention import attention_with_mask


# ---------------------------------------------------------------------------
# Custom autograd function: quantize Q/K/V forward; quantize their grads back
# ---------------------------------------------------------------------------
class _HIF8QKVQuantFunction(torch.autograd.Function):
    """
    Returns quantized (Q_q, K_q, V_q).  The actual attention call is made
    outside, allowing attention_with_mask to choose the backend (SDPA / FA3 / FA2).

    forward:
        Q_q = quant_dequant_hif8(Q * (smf / max|Q|)) / (smf / max|Q|)
        K_q, V_q – same

    backward (receives dL/dQ_q, dL/dK_q, dL/dV_q from attention autograd):
        return quant_dequant_hif8(grad * (smb / max|grad|)) / ...
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_max_forward: float,
        scale_max_backward: float,
    ):
        q_q, _ = _quant(q, scale_max_forward)
        k_q, _ = _quant(k, scale_max_forward)
        v_q, _ = _quant(v, scale_max_forward)

        ctx.save_for_backward(
            torch.tensor(scale_max_backward, dtype=q.dtype, device=q.device)
        )
        # Return 3 tensors; autograd tracks each separately
        return q_q, k_q, v_q

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        (smb,) = ctx.saved_tensors
        smb_val = smb.item()

        def _quant_grad(g):
            if g is None:
                return None
            g_q, _ = _quant(g, smb_val)
            return g_q

        return _quant_grad(grad_q), _quant_grad(grad_k), _quant_grad(grad_v), None, None


# ---------------------------------------------------------------------------
# Public API: drop-in replacement for attention_with_mask with HIF8 on Q/K/V
# ---------------------------------------------------------------------------
def hif8_attention_with_mask(
    q: torch.Tensor,            # [B, N, H, D]
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    attn_mask_kv: Optional[torch.Tensor] = None,
    is_cross_attn: bool = False,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    dropout_p: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
    scale_max_forward: float = 15.0,
    scale_max_backward: float = 224.0,
) -> torch.Tensor:
    """
    HIF8-quantized attention.  Quantizes Q, K, V before the kernel; quantizes
    dQ/dK/dV gradients in the backward pass.  All other kwargs are forwarded
    verbatim to attention_with_mask().
    """
    # --- Forward: inject quantization error into Q, K, V ---
    q_q, k_q, v_q = _HIF8QKVQuantFunction.apply(
        q, k, v, scale_max_forward, scale_max_backward
    )

    # --- Core attention: SDPA (NPU) / FA3 / FA2 (GPU) ---
    return attention_with_mask(
        q_q,
        k_q,
        v_q,
        attn_mask=attn_mask,
        attn_mask_kv=attn_mask_kv,
        is_cross_attn=is_cross_attn,
        causal=causal,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        dropout_p=dropout_p,
        dtype=dtype,
    )
