"""
不通过wrapper方式实现的cp，用于更复杂的cp场景。
"""
from typing import Optional
from torch.distributed import ProcessGroup
import torch.distributed as dist

class ContextParallelState:
    global_rank: int = 0
    # 全局cp group，等价于skiparse cp和context cp的并集
    global_cp_group: ProcessGroup = None
    global_cp_rank: int = 0
    global_cp_size: int = 1
    # Ulysses context parallel group
    cp_group: ProcessGroup = None
    cp_rank: int = 0
    cp_size: int = 1
    # skiparse context parallel group
    skiparse_cp_group: ProcessGroup = None
    skiparse_cp_rank: int = 0
    skiparse_cp_size: int = 1
    # 用于osp_next中full blocks的cp group
    full_cp_group: ProcessGroup = None
    full_cp_rank: int = 0
    full_cp_size: int = 1
    # 是否初始化cp state
    is_initialized: bool = False
    reset_counts: int = 0

    def log(self):
        if self.global_rank == 0:
            logs = "=" * 20 + f" CP State Reset (#{self.reset_counts}) " + "=" * 20
            logs += f"\nGlobal CP Group: {self.global_cp_group}"
            logs += f"\nGlobal CP Rank: {self.global_cp_rank}"
            logs += f"\nGlobal CP Size: {self.global_cp_size}"
            logs += f"\nCP Group: {self.cp_group}"
            logs += f"\nCP Rank: {self.cp_rank}"
            logs += f"\nCP Size: {self.cp_size}"
            logs += f"\nSkiparse CP Group: {self.skiparse_cp_group}"
            logs += f"\nSkiparse CP Rank: {self.skiparse_cp_rank}"
            logs += f"\nSkiparse CP Size: {self.skiparse_cp_size}"
            logs += f"\nFull CP Group: {self.full_cp_group}"
            logs += f"\nFull CP Rank: {self.full_cp_rank}"
            logs += f"\nFull CP Size: {self.full_cp_size}"
            logs += f"\n"
            logs += "=" * 20 + f" CP State Reset (#{self.reset_counts}) " + "=" * 20
            print(logs)

    def reset(
        self, 
        global_cp_group: ProcessGroup = None, 
        cp_group: ProcessGroup = None, 
        skiparse_cp_group: ProcessGroup = None,
        full_cp_group: ProcessGroup = None,
    ):
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        if global_cp_group is not None:
            self.global_cp_group = global_cp_group
            self.global_cp_rank = dist.get_rank(global_cp_group)
            self.global_cp_size = dist.get_world_size(global_cp_group)
        if cp_group is not None:
            self.cp_group = cp_group
            self.cp_rank = dist.get_rank(cp_group)
            self.cp_size = dist.get_world_size(cp_group)
        if skiparse_cp_group is not None:
            self.skiparse_cp_group = skiparse_cp_group
            self.skiparse_cp_rank = dist.get_rank(skiparse_cp_group)
            self.skiparse_cp_size = dist.get_world_size(skiparse_cp_group)
        if full_cp_group is not None:
            self.full_cp_group = full_cp_group
            self.full_cp_rank = dist.get_rank(full_cp_group)
            self.full_cp_size = dist.get_world_size(full_cp_group)
        self.is_initialized = True
        self.reset_counts += 1
        self.log()


    def clear(self):
        self.global_rank = 0
        self.global_cp_group = None
        self.global_cp_rank = 0
        self.global_cp_size = 1
        self.cp_group = None
        self.cp_rank = 0
        self.cp_size = 1
        self.skiparse_cp_group = None
        self.skiparse_cp_rank = 0
        self.skiparse_cp_size = 1
        self.full_cp_group = None
        self.full_cp_rank = 0
        self.full_cp_size = 1
        self.is_initialized = False

    def get_cp_infos_with_type(self, cp_type: Optional[str] = None):
        if cp_type is None:
            return self.cp_group, self.cp_rank, self.cp_size
        if cp_type == "cp":
            return self.cp_group, self.cp_rank, self.cp_size
        elif cp_type == "skiparse_cp":
            return self.skiparse_cp_group, self.skiparse_cp_rank, self.skiparse_cp_size
        elif cp_type == "full_blocks_cp":
            return self.full_cp_group, self.full_cp_rank, self.full_cp_size
        elif cp_type == "global_cp":
            return self.global_cp_group, self.global_cp_rank, self.global_cp_size
        else:
            raise ValueError(f"Invalid cp type: {cp_type}")

cp_state = ContextParallelState()

def use_context_parallel():
    return cp_state.is_initialized and cp_state.cp_size > 1

def use_skiparse_context_parallel():
    return cp_state.is_initialized and cp_state.skiparse_cp_size > 1

def use_full_blocks_context_parallel():
    return cp_state.is_initialized and cp_state.full_cp_size > 1