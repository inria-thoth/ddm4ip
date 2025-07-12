import dataclasses
import enum
from typing import Any, Generic, TypeVar
import typing

import deepinv
from omegaconf import DictConfig
import torch
import torch.utils.data

from ddm4ip.degradations.degradation import init_perturbation
from ddm4ip.utils import distributed


class Datasplit(enum.Enum):
    TRAIN = "train"
    TEST = "test"


@dataclasses.dataclass
class Batch:
    clean: torch.Tensor | None
    clean_label: torch.Tensor | None
    corrupt: torch.Tensor | None
    corrupt_label: torch.Tensor | None
    noise_level: torch.Tensor | None
    clean_conditioning: torch.Tensor | None
    corrupt_conditioning: torch.Tensor | None
    kernel: torch.Tensor | None = None
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def cuda(self):
        return self.to(device="cuda")

    @property
    def batch_size(self):
        if self.clean is not None:
            return self.clean.shape[0]
        elif self.corrupt is not None:
            return self.corrupt.shape[0]
        raise RuntimeError("Cannot figure out batch size")

    def apply(self, fn):
        non_meta_applied = {
            k: (fn(v) if v is not None else None) for k, v in vars(self).items() if k != "meta"
        }
        meta = self.meta
        meta_applied = {k: fn(v) for k, v in meta.items()}
        return Batch(
            **non_meta_applied,
            meta=meta_applied,
        )

    def __getitem__(self, index):
        return self.apply(lambda var: var[index])

    def to(self, device):
        return self.apply(lambda var: var.to(device))

    @staticmethod
    def collate_fn(batch):
        def collate_one(lst):
            if lst[0] is None:
                assert all(el is None for el in lst)
                out = None
            else:
                out = torch.utils.data.default_collate(lst)
            return out
        if len(batch) == 0:
            coll_meta = {}
        else:
            # TODO: Assert keys are always the same for whole batch
            coll_meta = {
                k: collate_one([el.meta[k] for el in batch])
                for k in batch[0].meta.keys()
            }
        return Batch(
            clean=collate_one([el.clean for el in batch]),
            clean_label=collate_one([el.clean_label for el in batch]),
            corrupt=collate_one([el.corrupt for el in batch]),
            corrupt_label=collate_one([el.corrupt_label for el in batch]),
            noise_level=collate_one([el.noise_level for el in batch]),
            clean_conditioning=collate_one([el.clean_conditioning for el in batch]),
            corrupt_conditioning=collate_one([el.corrupt_conditioning for el in batch]),
            kernel=collate_one([el.kernel for el in batch]),
            meta=coll_meta,
        )


def worker_init_fn(worker_id):
    from .image_folder_dataset import ZipImagePath
    # Needed: the cache must be different per-process
    ZipImagePath.clear_cache()


CORR_TYPE = TypeVar('CORR_TYPE', deepinv.physics.Physics, None)

class DatasetType(typing.Protocol, Generic[CORR_TYPE]):
    corruption: CORR_TYPE
    corrupt_img_size: tuple[int, int, int] = (0, 0, 0)
    corrupt_conditioning_channels: int = 0
    clean_img_size: tuple[int, int, int] = (0, 0, 0)
    clean_conditioning_channels: int = 0
    label_dim: int

    def __getitem__(self, index) -> Batch:
        ...

    def __len__(self) -> int:
        ...

    @property
    def noise_level(self) -> torch.Tensor | float:
        raise NotImplementedError


def init_dataloader(
    dset: torch.utils.data.Dataset[Batch],
    split: Datasplit,
    batch_size: int,
    num_workers: int,
    seed: int,
    start_idx: int,
    is_infinite: bool = True,
):
    from utils.torch_utils import InfiniteSampler
    mp_ctx = None
    prefetch_factor = None
    if num_workers > 0:
        # import multiprocess
        # mp_ctx = multiprocess.context.SpawnContext()
        # torch.utils.data.dataloader.python_multiprocessing = multiprocess # type: ignore
        prefetch_factor = 4
    if split == Datasplit.TRAIN:
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor,
            collate_fn=Batch.collate_fn,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=mp_ctx,
            sampler=InfiniteSampler(
                dataset=dset,
                rank=distributed.get_rank(),
                num_replicas=distributed.get_world_size(),
                shuffle=True,
                seed=seed,
                start_idx=start_idx,
                is_infinite=is_infinite,
            )
        )
    else:
        return torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=None,
            collate_fn=Batch.collate_fn,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=mp_ctx,
            drop_last=False,
            sampler=InfiniteSampler(
                dataset=dset,
                rank=distributed.get_rank(),
                num_replicas=distributed.get_world_size(),
                shuffle=False,
                start_idx=start_idx,
                is_infinite=is_infinite,
            )
        )


def init_dataset(cfg: DictConfig, split: Datasplit, is_paired: bool):
    """Support initializing dataset from two configuration styles:
    1. Specify both train/test datasets in a single configuration,
        so they will have the same degradation model. Must be under
        config['dataset']. The configuration must have separate
        'train_path' and 'test_path' keys to load the different splits.
    2. Specify train and test datasets in separate configurations
        under config['dataset']['train'] and config['dataset']['test'].
    Note that the test dataset is optional but the train dataset is mandatory.
    """
    dset_cfg = cfg["dataset"]
    path_keys = {
        Datasplit.TRAIN: "train_path",
        Datasplit.TEST: "test_path",
    }
    if split.value in dset_cfg:
        dset_cfg = dset_cfg[split.value]
        data_path = dset_cfg.get("path", dset_cfg.get(path_keys[split]))
    else:
        data_path = dset_cfg.get(path_keys[split])
    if data_path is None:
        raise RuntimeError(
            f"{split.name} dataset configuration has no path specified."
        )
    perturbation = init_perturbation(dset_cfg.degradation, dset_cfg.noise)

    dset_name = dset_cfg["name"]
    shuffle_clean = split == Datasplit.TRAIN and not is_paired
    if dset_name == 'zip':
        from .image_folder_dataset import ZipFileDataset
        return ZipFileDataset(
            data_path,
            degradation=perturbation,
            shuffle_clean=shuffle_clean,
            dset_cfg=dset_cfg,
            split=split,
        )
    elif dset_name == 'patch':
        from .patch_dataset import PatchDataset
        return PatchDataset(
            data_path,
            degradation=perturbation,
            shuffle_clean=shuffle_clean,
            dset_cfg=dset_cfg,
            split=split,
        )
    elif dset_name == 'simple_patch':
        from .simple_patch_dataset import SimplePatchDataset
        return SimplePatchDataset(
            data_path,
            degradation=perturbation,
            shuffle_clean=shuffle_clean,
            dset_cfg=dset_cfg,
            split=split,
        )
    else:
        raise ValueError(dset_name)
