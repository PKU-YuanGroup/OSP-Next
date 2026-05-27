import torch
import logging
from collections import OrderedDict
from typing import Hashable, Any
from accelerate.utils import is_npu_available as accelerate_is_npu_available

IS_NPU_AVAILABLE = None
# dtype encoding - decoding
DTYPE_TO_INT = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}

def str_to_precision(s):
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    elif s == "float16" or s == "fp16":
        return torch.float16
    elif s == "float32" or s == "float" or s == "fp32":
        return torch.float32
    elif s == "float64" or s == "double" or s == "fp64":
        return torch.float64
    elif s == "int64":
        return torch.int64
    elif s == "int32" or s == "int":
        return torch.int32
    elif s == "uint8":
        return torch.uint8
    else:
        raise ValueError(f"Unsupported precision string: {s}")

def precision_to_str(precision):
    if precision == torch.bfloat16:
        return "bfloat16"
    elif precision == torch.float16:
        return "float16"
    elif precision == torch.float32:
        return "float32"
    elif precision == torch.float64:
        return "float64"
    elif precision == torch.int64:
        return "int64"
    elif precision == torch.int32:
        return "int32"
    elif precision == torch.uint8:
        return "uint8"
    else:
        raise ValueError(f"Unsupported precision: {precision}")

def int_to_precision(int):
    if int in INT_TO_DTYPE:
        return INT_TO_DTYPE[int]
    else:
        raise ValueError(f"Unsupported precision int {int}")
    
def precision_to_int(precision):
    if precision in DTYPE_TO_INT:
        return DTYPE_TO_INT[precision]
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
def params_nums_to_str(params_num):
    if params_num >= 1e9:
        return f"{params_num / 1e9:.2f}B"
    elif params_num >= 1e6:
        return f"{params_num / 1e6:.2f}M"
    elif params_num >= 1e3:
        return f"{params_num / 1e3:.2f}K"
    else:
        return str(params_num)

def get_memory_allocated():
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f}"  # GiB

def is_npu_available():
    global IS_NPU_AVAILABLE
    if IS_NPU_AVAILABLE is None:
        IS_NPU_AVAILABLE = accelerate_is_npu_available(True)
    return IS_NPU_AVAILABLE

def check_and_import_npu():
    if is_npu_available():
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch_npu.npu.config.allow_internal_format = False

def safe_get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def contiguous(x):
    if x is None:
        return None
    return x.contiguous() if not x.is_contiguous() else x

class SafeCacheManager:

    DEFAULT_MAX_CACHE_SIZE = 1

    def __init__(self, max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[Hashable, Any] = OrderedDict()

    def is_exist(self, key: Hashable) -> bool:
        return key in self.cache

    def get(self, key: Hashable) -> Any:
        try:
            value = self.cache[key]
            self.cache.move_to_end(key, last=True)
            return value
        except KeyError:
            return None

    def _evict_if_needed(self):
        while len(self.cache) > self.max_cache_size > 0:
            self.cache.popitem(last=False)

    def set(self, key: Hashable, value: Any):
        self.cache[key] = value
        self.cache.move_to_end(key, last=True)
        self._evict_if_needed()

    def clear(self):
        self.cache.clear()