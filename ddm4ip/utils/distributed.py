# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
from typing import Any, List, Optional
from socket import gethostname
import torch
import torch.distributed
from torch.nn import functional as F

from . import training_stats


_sync_device = None

#----------------------------------------------------------------------------


def fetch_env_var(*var_names) -> str:
    for name in var_names:
        try:
            return os.environ[name]
        except KeyError:
            continue
    raise KeyError(f"No variable among {var_names} could be fetched.")


def init(do_init: bool = True) -> int:
    global _sync_device

    if not do_init:
        if torch.cuda.device_count() != 1:
            print("Distributed training must be initialized when multiple devices "
                  "are present, but `do_init` was set to False.")
            raise RuntimeError("Distributed training must be initialized when multiple devices are present")
        print("Distributed training will not be initialized.")
        torch.cuda.set_device(0)
        return 0

    try:
        master_addr = fetch_env_var("MASTER_ADDR")
        master_port = fetch_env_var("MASTER_PORT")
        rank = int(fetch_env_var("RANK", "SLURM_PROCID"))
        local_rank = int(fetch_env_var("LOCAL_RANK", "SLURM_LOCALID"))
        world_size = int(fetch_env_var("WORLD_SIZE"))
    except KeyError as e:
        print(f"Falling back to single GPU mode. Error: {e}")
        master_addr = 'localhost'
        master_port = '29512'
        rank = 0
        local_rank = 0
        world_size = 1

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    if world_size > 1:
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        print(f"Distributed initialization at rank {rank} of {world_size} "
              f"(rank {local_rank} on {gethostname()} with {torch.cuda.device_count()} GPUs allocated).")
        torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(local_rank)
    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)

    return local_rank


def get_local_rank():
    if torch.distributed.is_initialized():
        return int(fetch_env_var("LOCAL_RANK", "SLURM_LOCALID"))
    return 0


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def broadcast_obj(obj, src=0):
    if torch.distributed.is_initialized():
        to_broadcast = [obj]
        torch.distributed.broadcast_object_list(to_broadcast, src=src)
        return to_broadcast[0]
    return obj

#----------------------------------------------------------------------------


def _simple_gather_all_tensors(result: torch.Tensor, group: Any, world_size: int) -> List[torch.Tensor]:
    with torch.no_grad():
        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_result, result, group)
    # to propagate autograd graph from local rank
    gathered_result[torch.distributed.get_rank(group)] = result
    return gathered_result


def gather_all_tensors_opt(tensor: torch.Tensor | None, group: Optional[Any] = None) -> List[torch.Tensor]:
    if not torch.distributed.is_initialized():
        return [tensor] if tensor is not None else []

    # If tensor is None, set the first dimension to 0 and the rest to the same dimensions as another
    # tensor in the group.
    shapes = []
    tensor_list = []
    for r in range(get_world_size()):
        if r == get_rank() and tensor is not None:
            c_shape = tensor.shape
        else:
            c_shape = None
        shapes.append(broadcast_obj(c_shape, src=r))
    try:
        default_shape = list(next(s for s in shapes if s is not None))
        default_shape[0] = 0
    except StopIteration:
        # All tensors are None
        return []

    if tensor is None:
        tensor = torch.empty(default_shape)  # default dtype, device (and hope it's correct!)

    tensor_list = gather_all_tensors(tensor, group)
    tensor_list = [t for i, t in enumerate(tensor_list) if shapes[i] is not None]
    return tensor_list


def gather_all_tensors(result: torch.Tensor, group: Optional[Any] = None) -> List[torch.Tensor]:
    """Gather all tensors from several ddp processes onto a list that is broadcast to all processes.

    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        list with size equal to the process group where element i corresponds to result tensor from process i

    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    with torch.no_grad():
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = F.pad(result, pad_dims)
        gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][slice_param]
    # to propagate autograd graph from local rank
    gathered_result[torch.distributed.get_rank(group)] = result
    return gathered_result