# HIF8 quantization-aware linear layer for DiT (OSP-Next, NPU path)
#
# Simulates HIF8 low-precision quantization error on BOTH input activations
# AND weight matrices, in both the forward and backward passes.
#
# Forward:
#   scale_x = scale_max_forward / max(|x|)
#   x_q     = quant_dequant_float(x * scale_x) / scale_x
#
#   scale_w = scale_max_forward / max(|W|)
#   w_q     = quant_dequant_float(W * scale_w) / scale_w
#
#   out     = x_q @ w_q.T + bias          (both operands are dequantized)
#
# Backward:
#   scale_g  = scale_max_backward / max(|grad_out|)
#   grad_q   = quant_dequant_float(grad_out * scale_g) / scale_g
#
#   grad_x = grad_q @ w_q                 (use dequant W saved from forward)
#   grad_W = grad_q.T @ x_q              (use dequant x saved from forward)
#   grad_b = grad_q.sum(over batch dims)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# NPU operator import (placeholder until the actual op package is available)
# ---------------------------------------------------------------------------
try:
    from ospnext.quant_cy_npu import QType, quant_dequant_float  # Requires building the quant_cy_npu kernel first; build steps: unzip, then run `bash build.sh`.
except ImportError:
    # Temporary stub so the code can be imported without the real op.
    # Replace with the real import once hif8_ops is installed.
    def quant_dequant_float(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "quant_dequant_float is not available. "
            "Please install the hif8_ops NPU package."
        )


def _quant(x: torch.Tensor, scale_max: float) -> tuple:
    """Compute per-tensor scale and return (dequantized_x, scale)."""
    qtype = QType("hif8")
    x_max = x.abs().amax().clamp(min=1e-8)                                  
    scale = scale_max / x_max
    scaled_x = x * scale
    x_q = quant_dequant_float(scaled_x, qtype, force_fp32=True) / scale      # Final quantize-dequantize'd tensor, used by downstream linear or attention ops.
    # quant_dequant_float returns FP32; cast with .to(torch.bfloat16) downstream when needed.
    return x_q, scale


# ---------------------------------------------------------------------------
# Custom autograd function: quantizes x AND W in forward, grad in backward
# ---------------------------------------------------------------------------
class _HIF8LinearFunction(torch.autograd.Function):
    """
    Unified autograd function for HIF8-quantized linear layer.

    Inputs to apply():
        x                  – input activation  (..., in_features)
        weight             – weight matrix      (out_features, in_features)
        bias               – bias vector or None
        scale_max_forward  – HIF8 max for activations and weight (e.g. 15)
        scale_max_backward – HIF8 max for gradients              (e.g. 224)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, scale_max_forward: float, scale_max_backward: float):
        # --- Quantize input ---
        x_q, _ = _quant(x, scale_max_forward)

        # --- Quantize weight ---
        w_q, _ = _quant(weight, scale_max_forward)
        # w_q, _ = _quant(weight, scale_max_forward).to(torch.bfloat16) # optional

        # Save dequantized tensors for backward
        ctx.save_for_backward(
            x_q,
            w_q,
            torch.tensor(scale_max_backward, dtype=x.dtype, device=x.device),
        )
        ctx.has_bias = bias is not None

        # out = dequant(x) @ dequant(W).T + bias
        return F.linear(x_q, w_q, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x_q, w_q, smb_t = ctx.saved_tensors
        scale_max_backward = smb_t.item()

        # --- Quantize incoming gradient ---
        grad_q, _ = _quant(grad_output, scale_max_backward)

        # --- grad w.r.t. x: grad_q @ w_q ---
        # grad_q: (..., out_features)   w_q: (out_features, in_features)
        grad_input = grad_q @ w_q   # (..., in_features) – shape matches x

        # --- grad w.r.t. W: flatten batch dims, then outer product ---
        # x_q:   (..., in_features)  →  (M, in_features)
        # grad_q: (..., out_features) →  (M, out_features)
        x_2d = x_q.reshape(-1, x_q.shape[-1])    # (M, in_features)
        g_2d = grad_q.reshape(-1, grad_q.shape[-1])  # (M, out_features)
        grad_weight = g_2d.t() @ x_2d            # (out_features, in_features)

        # --- grad w.r.t. bias ---
        grad_bias = g_2d.sum(0) if ctx.has_bias else None

        # Return grads for: x, weight, bias, scale_max_forward, scale_max_backward
        return grad_input, grad_weight, grad_bias, None, None


# ---------------------------------------------------------------------------
# HIF8Linear: drop-in nn.Linear replacement
# ---------------------------------------------------------------------------
class HIF8Linear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with HIF8 quant-dequant simulation.

    Both the input activation and the weight matrix are quantized to HIF8
    before the matrix multiplication.  The backward pass quantizes the
    incoming gradient as well.

    Forward computation:
        x_q  = dequant( quant_hif8( x  * (scale_max_forward  / max|x|)  ) )
        w_q  = dequant( quant_hif8( W  * (scale_max_forward  / max|W|)  ) )
        out  = x_q @ w_q.T + bias

    Backward computation:
        g_q  = dequant( quant_hif8( g  * (scale_max_backward / max|g|)  ) )
        ∂x   = g_q  @ w_q
        ∂W   = g_q.T @ x_q
        ∂b   = g_q.sum(batch)

    Args:
        in_features:        Same as nn.Linear.
        out_features:       Same as nn.Linear.
        bias:               Same as nn.Linear.
        scale_max_forward:  HIF8 representable max for activations/weights (default 15).
        scale_max_backward: HIF8 representable max for gradients            (default 224).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_max_forward: float = 15.0,
        scale_max_backward: float = 224.0,
    ):
        super().__init__(in_features, out_features, bias)
        self.scale_max_forward = scale_max_forward
        self.scale_max_backward = scale_max_backward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HIF8LinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.scale_max_forward,
            self.scale_max_backward,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"scale_max_fwd={self.scale_max_forward}, "
            f"scale_max_bwd={self.scale_max_backward}"
        )
