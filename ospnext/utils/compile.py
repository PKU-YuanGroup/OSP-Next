import torch
from ospnext.utils.utils import is_npu_available

if not is_npu_available():
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.accumulated_cache_size_limit = 32

def maybe_compile(disable=False):
    def decorator(func):
        if is_npu_available():
            return func
        if disable:
            return torch.compiler.disable(func)
        return torch.compile(func)
    return decorator