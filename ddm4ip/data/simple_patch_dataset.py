from typing import Tuple

import torch
import torchvision.transforms.v2 as v2
import deepinv

from ddm4ip.data.base import Batch, DatasetType, Datasplit
from ddm4ip.data.image_folder_dataset import BaseImageFolderDataset
from ddm4ip.data.utils import  SeededTransform, TopLeftCrop
from ddm4ip.utils import distributed


def inflate_patch_size(patch_size: int, filter_size: int) -> int:
    pad_before, pad_after = 0, 0
    if filter_size > 0:
        pad_before = filter_size // 2
        pad_after = filter_size // 2 - (filter_size - 1) % 2
    return patch_size + pad_before + pad_after


class SimplePatchDataset(torch.utils.data.Dataset[Batch], DatasetType):
    def __init__(
        self,
        path,
        degradation: deepinv.physics.Physics | None,
        split: Datasplit,
        dset_cfg,
        shuffle_clean: bool = True,
        generator=None,
    ):
        self.clean_path = path
        self.noisy_path = dset_cfg["noisy_path"]
        self.split = split
        self.shuffle_clean = shuffle_clean
        self.patch_size = dset_cfg["patch_size"]
        self.x_flip = dset_cfg.get("x_flip", False) and self.split == Datasplit.TRAIN
        self.space_conditioning = dset_cfg.get("space_conditioning", False)
        self.inflate_patches = dset_cfg.get("inflate_patches", 0)
        assert degradation is None
        self.init_full_datasets(dset_cfg, generator)
        self.dset_length = max(len(self.clean_full_data), len(self.noisy_full_data))
        self.clean_transform, self.noisy_transform = self.init_transforms()

        # Run the pipeline once to get a sample
        tst_b = self[0]
        assert tst_b.corrupt is not None and tst_b.clean is not None
        self.corrupt_img_size = (tst_b.corrupt.shape[-3], tst_b.corrupt.shape[-2], tst_b.corrupt.shape[-1])
        self.corrupt_conditioning_channels = tst_b.corrupt_conditioning.shape[-3] if tst_b.corrupt_conditioning is not None else 0
        self.clean_img_size = (tst_b.clean.shape[-3], tst_b.clean.shape[-2], tst_b.clean.shape[-1])
        self.clean_conditioning_channels = tst_b.clean_conditioning.shape[-3] if tst_b.clean_conditioning is not None else 0
        self.label_dim = 0
        self.corruption = None

        distributed.print0(f"Loaded SimplePatchDataset dataset at '{path}':")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Inflate patches (k-size):       {self.inflate_patches}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Number of base noisy images:    {len(self.noisy_full_ids)}")
        distributed.print0(f"Number of base clean images:    {len(self.clean_full_ids)}")
        distributed.print0()

    def init_transforms(self):
        img_transform: list[v2.Transform] = []
        act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
        if self.split == Datasplit.TRAIN:
            img_transform.append(v2.RandomCrop(act_patch_size, pad_if_needed=True))
        else:
            img_transform.append(TopLeftCrop(act_patch_size))
        if self.x_flip:
            img_transform.append(v2.RandomHorizontalFlip(p=0.5))

        clean_seed = int(torch.randint(0, 1_000_000_000, size=(1,)).item())
        noisy_seed = int(torch.randint(0, 1_000_000_000, size=(1,)).item()) if self.shuffle_clean else clean_seed

        clean_trsf = SeededTransform(clean_seed, v2.Compose(img_transform))
        noisy_trsf = SeededTransform(noisy_seed, v2.Compose(img_transform))
        return clean_trsf, noisy_trsf

    def init_full_datasets(self, cfg, rnd_gen):
        full_img_trsf = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        self.clean_full_data = BaseImageFolderDataset(
            path=self.clean_path,
            cache=False,
            use_labels=False,
            img_transform=v2.Compose(full_img_trsf),
            max_imgs=None,
            filter=cfg.get("clean_filter", None)
        )
        self.noisy_full_data = BaseImageFolderDataset(
            path=self.noisy_path or self.clean_path,
            cache=False,
            use_labels=False,
            img_transform=v2.Compose(full_img_trsf),
            max_imgs=None,
            filter=cfg.get("noisy_filter", None)
        )
        self.noisy_full_ids = torch.arange(len(self.noisy_full_data))
        if self.shuffle_clean:
            self.clean_full_ids = torch.randperm(len(self.clean_full_data), generator=rnd_gen)
        else:
            self.clean_full_ids = torch.arange(len(self.clean_full_data))

    def get_conditioning(self, img: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        cond = None
        if self.space_conditioning and img is not None:
            cond = img[..., -2:, :, :]
            img = img[..., :-2, :, :]
        return cond, img

    def __getitem__(self, idx):
        clean_idx = int(self.clean_full_ids[idx % len(self.clean_full_ids)])
        clean_imgwcond, clean_tgt = self.clean_full_data[clean_idx]
        clean_imgwcond = self.clean_transform(clean_imgwcond)
        clean_cond, clean_img = self.get_conditioning(clean_imgwcond)

        noisy_idx = int(self.noisy_full_ids[idx % len(self.noisy_full_ids)])
        noisy_imgwcond, noisy_tgt = self.noisy_full_data[noisy_idx]
        noisy_imgwcond = self.noisy_transform(noisy_imgwcond)
        noisy_cond, noisy_img = self.get_conditioning(noisy_imgwcond)

        return Batch(
            clean=clean_img,
            corrupt=noisy_img,
            clean_label=None,
            corrupt_label=None,
            noise_level=self.noise_level,
            clean_conditioning=clean_cond,
            corrupt_conditioning=noisy_cond,
        )

    def __len__(self):
        return self.dset_length

    @property
    def noise_level(self) -> torch.Tensor:
        return torch.tensor(0.0)
