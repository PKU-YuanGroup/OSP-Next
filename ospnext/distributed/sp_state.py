from typing import Optional
from torch.distributed import ProcessGroup
import torch.distributed as dist

class SequenceParallelState:
    global_rank: int = 0
    # Global SP group, equivalent to the union of skiparse_sp and sp groups
    global_sp_group: ProcessGroup = None
    global_sp_rank: int = 0
    global_sp_size: int = 1
    # Ulysses sequence parallel group
    sp_group: ProcessGroup = None
    sp_rank: int = 0
    sp_size: int = 1
    # skiparse sequence parallel group
    skiparse_sp_group: ProcessGroup = None
    skiparse_sp_rank: int = 0
    skiparse_sp_size: int = 1
    # SP group used by the full blocks in osp_next
    full_sp_group: ProcessGroup = None
    full_sp_rank: int = 0
    full_sp_size: int = 1
    # Whether the SP state has been initialized
    is_initialized: bool = False
    reset_counts: int = 0

    def log(self):
        if self.global_rank == 0:
            logs = "=" * 20 + f" SP State Reset (#{self.reset_counts}) " + "=" * 20
            logs += f"\nGlobal SP Group: {self.global_sp_group}"
            logs += f"\nGlobal SP Rank: {self.global_sp_rank}"
            logs += f"\nGlobal SP Size: {self.global_sp_size}"
            logs += f"\nSP Group: {self.sp_group}"
            logs += f"\nSP Rank: {self.sp_rank}"
            logs += f"\nSP Size: {self.sp_size}"
            logs += f"\nSkiparse SP Group: {self.skiparse_sp_group}"
            logs += f"\nSkiparse SP Rank: {self.skiparse_sp_rank}"
            logs += f"\nSkiparse SP Size: {self.skiparse_sp_size}"
            logs += f"\nFull SP Group: {self.full_sp_group}"
            logs += f"\nFull SP Rank: {self.full_sp_rank}"
            logs += f"\nFull SP Size: {self.full_sp_size}"
            logs += f"\n"
            logs += "=" * 20 + f" SP State Reset (#{self.reset_counts}) " + "=" * 20
            print(logs)

    def reset(
        self, 
        global_sp_group: ProcessGroup = None, 
        sp_group: ProcessGroup = None, 
        skiparse_sp_group: ProcessGroup = None,
        full_sp_group: ProcessGroup = None,
    ):
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        if global_sp_group is not None:
            self.global_sp_group = global_sp_group
            self.global_sp_rank = dist.get_rank(global_sp_group)
            self.global_sp_size = dist.get_world_size(global_sp_group)
        if sp_group is not None:
            self.sp_group = sp_group
            self.sp_rank = dist.get_rank(sp_group)
            self.sp_size = dist.get_world_size(sp_group)
        if skiparse_sp_group is not None:
            self.skiparse_sp_group = skiparse_sp_group
            self.skiparse_sp_rank = dist.get_rank(skiparse_sp_group)
            self.skiparse_sp_size = dist.get_world_size(skiparse_sp_group)
        if full_sp_group is not None:
            self.full_sp_group = full_sp_group
            self.full_sp_rank = dist.get_rank(full_sp_group)
            self.full_sp_size = dist.get_world_size(full_sp_group)
        self.is_initialized = True
        self.reset_counts += 1
        self.log()


    def clear(self):
        self.global_rank = 0
        self.global_sp_group = None
        self.global_sp_rank = 0
        self.global_sp_size = 1
        self.sp_group = None
        self.sp_rank = 0
        self.sp_size = 1
        self.skiparse_sp_group = None
        self.skiparse_sp_rank = 0
        self.skiparse_sp_size = 1
        self.full_sp_group = None
        self.full_sp_rank = 0
        self.full_sp_size = 1
        self.is_initialized = False

    def get_sp_infos_with_type(self, sp_type: Optional[str] = None):
        if sp_type is None:
            return self.sp_group, self.sp_rank, self.sp_size
        if sp_type == "sp":
            return self.sp_group, self.sp_rank, self.sp_size
        elif sp_type == "skiparse_sp":
            return self.skiparse_sp_group, self.skiparse_sp_rank, self.skiparse_sp_size
        elif sp_type == "full_blocks_sp":
            return self.full_sp_group, self.full_sp_rank, self.full_sp_size
        elif sp_type == "global_sp":
            return self.global_sp_group, self.global_sp_rank, self.global_sp_size
        else:
            raise ValueError(f"Invalid sp type: {sp_type}")

sp_state = SequenceParallelState()

def use_sequence_parallel():
    return sp_state.is_initialized and sp_state.sp_size > 1

def use_skiparse_sequence_parallel():
    return sp_state.is_initialized and sp_state.skiparse_sp_size > 1

def use_full_blocks_sequence_parallel():
    return sp_state.is_initialized and sp_state.full_sp_size > 1