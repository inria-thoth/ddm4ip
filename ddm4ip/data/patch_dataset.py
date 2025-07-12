from functools import partial
import gc
from typing import Tuple, TypeVar

import numpy as np
import torch
import torchvision.transforms.v2 as v2
import deepinv

from ddm4ip.data.base import Batch, DatasetType, Datasplit
from ddm4ip.data.image_folder_dataset import BaseImageFolderDataset
from ddm4ip.data.utils import (
    AddLocMapTransform,
    AddRandomLocMapTransform,
    GetAllPatchesTransform,
    ReplaceRandomLocMapTransform,
    get_dim_blocks,
)
from ddm4ip.utils.torch_utils import crop_valid
from ddm4ip.utils import distributed


def inflate_patch_size(patch_size: int, filter_size: int) -> int:
    pad_before, pad_after = 0, 0
    if filter_size > 0:
        pad_before = filter_size // 2
        pad_after = filter_size // 2 - (filter_size - 1) % 2
    return patch_size + pad_before + pad_after


def split_list(lst, size):
    # behaves the same as torch.split
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


OPT_TEN_TYPE = TypeVar("OPT_TEN_TYPE", torch.Tensor, None)


def search_idx_in_list(all_num_patches: list[int], idx: int) -> tuple[int, int]:
    cs = np.cumsum(all_num_patches)
    full_img_idx: int = np.searchsorted(cs, idx, side="right") # type: ignore
    if full_img_idx == 0:
        remainder = idx
    elif full_img_idx == len(cs):
        raise IndexError(f"Index {idx} invalid.")
    else:
        remainder = idx - cs[full_img_idx - 1]
    return full_img_idx, remainder


class PatchDataset(torch.utils.data.Dataset[Batch], DatasetType):
    """Clean and corrupt (potentially unpaired) images"""
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
        self.noisy_path = dset_cfg.get("noisy_path", None)
        self.split = split
        self.shuffle_clean = shuffle_clean
        self.corruption: deepinv.physics.Physics | None = degradation
        self.patch_size = dset_cfg["patch_size"]
        self.x_flip = dset_cfg.get("x_flip", False) and self.split == Datasplit.TRAIN
        self.get_all_test_patches = dset_cfg.get("full_test", False)

        self.space_conditioning = dset_cfg.get("space_conditioning", False)
        self.random_space_conditioning = dset_cfg.get("random_space_conditioning", False)
        if self.space_conditioning and self.random_space_conditioning:
            raise ValueError("Space conditioning and random space conditioning cannot both be active.")
        self.random_replace_locmap = float(dset_cfg.get("random_replace_locmap", 0.0))
        if self.random_replace_locmap > 0 and not self.space_conditioning:
            raise ValueError("Cannot do random locmap replacements unless `space_conditioning` is True.")

        # inflate_patches: increase the actual patch-size due to cropping which occurs
        #                  when applying degradation.
        self.inflate_patches = dset_cfg.get("inflate_patches", 0)
        self.use_cuda = dset_cfg.get("cuda", False)
        self.need_clean: bool = dset_cfg.get("need_clean", True)
        self.need_noisy: bool = dset_cfg.get("need_noisy", True)
        assert self.need_clean or self.need_noisy

        self.init_full_datasets(dset_cfg, generator)

        # We can either use the patch cache - better randomization for training
        # or just iterate through the images and the patches within each image, without caching.
        self.patch_cache: dict[int, tuple[torch.Tensor | None, torch.Tensor | None] | None] = {}
        if self.split == Datasplit.TRAIN:
            self.num_patches_per_image = dset_cfg["num_patches_per_image"]
            self.patch_cache_size = dset_cfg.get("patch_cache_size", 2048)
            self.dset_length = self.patch_cache_size
        else:
            self.patch_cache_size = -1  # don't use zero to avoid risking divideByZero
            if self.get_all_test_patches:
                self.all_num_patches = [self.get_num_patches(full_img) for full_img, _ in self.clean_full_data]
                self.dset_length = sum(self.all_num_patches)
                self.num_patches_per_image = -1  # This is never used
            else:
                # Center-crop of each image
                self.dset_length = len(self.clean_full_data)
                self.num_patches_per_image = 1

        if self.use_cuda:
            self.device = torch.device("cuda", distributed.get_local_rank())
        else:
            self.device = torch.device("cpu")
        if self.corruption is not None:
            self.corruption = self.corruption.to(self.device)

        # Run the pipeline once to get a sample
        clean_patches, noisy_patches = self.get_next_patches(idx=0, num_patches=1, get_clean=True, get_noisy=True)
        assert clean_patches is not None and noisy_patches is not None
        # Record sample shapes
        noisy_cond, noisy_patch = self.get_conditioning(noisy_patches[0])
        self.corrupt_img_size = (noisy_patch.shape[-3], noisy_patch.shape[-2], noisy_patch.shape[-1])
        self.corrupt_conditioning_channels = noisy_cond.shape[-3] if noisy_cond is not None else 0
        clean_cond, clean_patch = self.get_conditioning(clean_patches[0])
        self.clean_img_size = (clean_patch.shape[-3], clean_patch.shape[-2], clean_patch.shape[-1])
        self.clean_conditioning_channels = clean_cond.shape[-3] if clean_cond is not None else 0
        self.label_dim = 0
        distributed.print0(f"Loaded PatchDataset dataset at '{path}':")
        if self.clean_full_data != self.noisy_full_data:
            distributed.print0("Clean and noisy datasets are different!")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Inflate patches (k-size):       {self.inflate_patches}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Space conditioning:             {'normal' if self.space_conditioning else 'random' if self.random_space_conditioning else 'none'}")
        distributed.print0(f"Random replace conditioning p:  {self.random_replace_locmap}")
        distributed.print0(f"Number of base images:          {len(self.clean_full_data)}")
        if self.split == Datasplit.TEST:
            distributed.print0(f"Full test patches:              {self.get_all_test_patches}")
            distributed.print0(f"Dataset length:                 {self.dset_length}")
        else:
            distributed.print0(f"Number patches per image:       {self.num_patches_per_image}")
            distributed.print0(f"Patch cache size:               {self.patch_cache_size}")
        distributed.print0()

    def init_full_datasets(self, cfg, rnd_gen):
        full_img_trsf = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if self.space_conditioning:
            full_img_trsf.append(AddLocMapTransform())
        clean_filter = cfg.get("clean_filter", None)
        self.clean_full_data = BaseImageFolderDataset(
            path=self.clean_path,
            cache=False,
            use_labels=False,
            img_transform=v2.Compose(full_img_trsf),
            max_imgs=None,
            filter=clean_filter
        )
        noisy_filter = cfg.get("noisy_filter", None)
        needs_separate_noisy = (
            (self.noisy_path is not None and self.noisy_path != self.clean_path)
            or (noisy_filter is not None and noisy_filter != clean_filter)
        )
        if needs_separate_noisy:
            self.noisy_full_data = BaseImageFolderDataset(
                path=self.noisy_path or self.clean_path,
                cache=False,
                use_labels=False,
                img_transform=v2.Compose(full_img_trsf),
                max_imgs=None,
                filter=noisy_filter
            )
        else:
            self.noisy_full_data = self.clean_full_data
        if len(self.clean_full_data) != len(self.noisy_full_data):
            raise ValueError(
                f"Clean and noisy datasets must have same length but found "
                f"{len(self.clean_full_data)} and {len(self.noisy_full_data)}"
            )
        # single `full_idx` since clean and noisy datasets are of equal length
        self.full_idx = 0
        self.noisy_full_ids = torch.arange(len(self.noisy_full_data))
        if self.shuffle_clean:
            self.clean_full_ids = torch.randperm(len(self.clean_full_data), generator=rnd_gen)
        else:
            self.clean_full_ids = torch.arange(len(self.clean_full_data))

    def get_num_patches(self, img: torch.Tensor):
        act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
        num_img_patches_h = get_dim_blocks(
            img.shape[-2], act_patch_size, stride=self.patch_size
        )
        num_img_patches_w = get_dim_blocks(
            img.shape[-1], act_patch_size, stride=self.patch_size
        )
        return num_img_patches_h * num_img_patches_w

    def corrupt_trsf(self, img: torch.Tensor) -> torch.Tensor:
        if self.corruption is None:
            return img
        conditioning, pure_img = self.get_conditioning(img)
        squeeze_out = False
        if pure_img.dim() == 3:
            squeeze_out = True
            pure_img = pure_img[None, ...]
        if conditioning is not None and conditioning.dim() == 3:
            conditioning = conditioning[None, ...]
        corrupt_img = self.corruption(
            pure_img, conditioning=conditioning,
        ).clamp(0, 1)
        full_img = self.add_conditioning(corrupt_img, conditioning)
        if squeeze_out:
            return full_img.squeeze(0)
        return full_img

    def get_patch_transform_func(self, num_patches: int, full_img_size: tuple[int, int]):
        img_transform: list[v2.Transform] = []
        act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
        if self.split == Datasplit.TRAIN:
            img_transform.append(v2.RandomCrop(act_patch_size, pad_if_needed=True))
            if self.random_replace_locmap > 0:
                img_transform.append(ReplaceRandomLocMapTransform(
                    self.random_replace_locmap, full_img_size
                ))
        else:
            if self.get_all_test_patches:
                img_transform.append(GetAllPatchesTransform(act_patch_size, stride=self.patch_size))
            else:
                img_transform.append(v2.CenterCrop(act_patch_size))
        if self.x_flip:
            img_transform.append(v2.RandomHorizontalFlip(p=0.5))

        if self.random_space_conditioning is not False:
            img_transform.append(AddRandomLocMapTransform(full_img_size))

        img_transform.append(v2.Lambda(lambda x: x.to(device=self.device)))

        # Seeds must be equal if paired, different if unpaired.
        # clean_seed = int(torch.randint(0, 1_000_000_000, size=(1,)).item())
        # noisy_seed = int(torch.randint(0, 1_000_000_000, size=(1,)).item()) if self.shuffle_clean else clean_seed

        trsf = v2.Compose(img_transform)#SeededTransform(clean_seed, v2.Compose(img_transform))
        # noisy_trsf = v2.Compose(img_transform)#SeededTransform(noisy_seed, v2.Compose(img_transform))
        if self.split == Datasplit.TEST and self.get_all_test_patches:
            # num_patches is ignored in this case
            def apply_unbind(img, trsf):
                return trsf(img).unbind(0)
            return partial(apply_unbind, trsf=trsf)#, partial(apply_unbind, trsf=noisy_trsf)
        else:
            def apply_multiple(img, trsf):
                return [trsf(img) for _ in range(num_patches)]
            return partial(apply_multiple, trsf=trsf)#, partial(apply_multiple, trsf=noisy_trsf)

    def get_next_patches(
        self,
        idx: int,
        num_patches: int | None = None,
        get_clean: bool | None = None,
        get_noisy: bool | None = None,
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        get_clean = get_clean if get_clean is not None else self.need_clean
        get_noisy = get_noisy if get_noisy is not None else self.need_noisy
        if num_patches is None:
            num_patches = self.num_patches_per_image

        clean_patches, clean_img = None, None
        if get_clean:
            clean_img, _ = self.clean_full_data[self.clean_full_ids[idx]]
            clean_patch_trsf = self.get_patch_transform_func(
                num_patches=num_patches, full_img_size=(clean_img.shape[-2], clean_img.shape[-1])
            )
            clean_patches = clean_patch_trsf(clean_img)

        noisy_patches, noisy_img = None, None
        if get_noisy:
            noisy_img, _ = self.noisy_full_data[self.noisy_full_ids[idx]]
            noisy_patch_trsf = self.get_patch_transform_func(
                num_patches=num_patches, full_img_size=(noisy_img.shape[-2], noisy_img.shape[-1])
            )
            noisy_patches = noisy_patch_trsf(noisy_img)
            noisy_patches = [self.corrupt_trsf(np) for np in noisy_patches]
        if clean_patches is not None and noisy_patches is not None:
            assert clean_img is not None and noisy_img is not None
            if len(clean_patches) != len(noisy_patches):
                raise ValueError(
                    f"Found {len(clean_patches)} clean and {len(noisy_patches)} noisy "
                    f"patches at image ID {idx}. Base images are of shape "
                    f"{clean_img.shape} and {noisy_img.shape}."
                )
        return clean_patches, noisy_patches

    def populate_test_patch_cache(self):
        clean_patches, noisy_patches = self.get_next_patches(idx=self.full_idx)
        self.patch_cache_size = len(clean_patches) if clean_patches is not None else len(noisy_patches) # type: ignore
        for j in range(self.patch_cache_size):
            cp = clean_patches[j] if clean_patches is not None else None
            np = noisy_patches[j] if noisy_patches is not None else None
            self.patch_cache[j] = (cp, np)

    def populate_train_patch_cache(self):
        keys = {k for k, v in self.patch_cache.items() if v is not None}
        missing_ids = list(set(range(self.patch_cache_size)) - keys)
        for i in range(0, len(missing_ids), self.num_patches_per_image):
            clean_patches, noisy_patches = self.get_next_patches(
                idx=self.full_idx,
                num_patches=min(self.num_patches_per_image, len(missing_ids))
            )
            num_patches = len(clean_patches) if clean_patches is not None else len(noisy_patches) # type: ignore
            for j in range(num_patches):
                cp = clean_patches[j] if clean_patches is not None else None
                np = noisy_patches[j] if noisy_patches is not None else None
                try:
                    self.patch_cache[missing_ids[i + j]] = (cp, np)
                except IndexError:
                    break
            # increment full_idx to fetch next full image. clean and noisy full data have same length.
            self.full_idx = (self.full_idx + 1) % len(self.clean_full_data)

    def get_conditioning(self, img: OPT_TEN_TYPE) -> Tuple[torch.Tensor | None, OPT_TEN_TYPE]:
        cond = []
        # noise conditioning is added after space conditioning
        if (self.space_conditioning or self.random_space_conditioning is not False) and img is not None:
            cond.append(img[..., -2:, :, :])
            img = img[..., :-2, :, :]
        if len(cond) == 0:
            cond = None
        elif len(cond) == 1:
            cond = cond[0]
        else:
            cond = torch.cat(cond[::-1], dim=-3)
        return cond, img

    def add_conditioning(self, img: torch.Tensor, conditioning: torch.Tensor | None) -> torch.Tensor:
        if conditioning is None:
            return img
        if img.shape[-1] == conditioning.shape[-1] and img.shape[-2] == conditioning.shape[-2]:
            return torch.cat((img, conditioning), dim=-3)
        elif self.inflate_patches > 0:
            return torch.cat((img, crop_valid(conditioning, self.inflate_patches)), dim=-3)
        raise RuntimeError(f"Cannot add conditioning of shape {conditioning.shape} to image of shape {img.shape}")

    def __getitem__(self, idx):
        if self.split == Datasplit.TEST:
            if self.get_all_test_patches:
                full_img_idx, cache_idx = search_idx_in_list(self.all_num_patches, idx)
            else:
                full_img_idx, cache_idx = idx, 0
            # clear cache if the index requested is not the same as the index of the cache
            if self.full_idx != full_img_idx:
                self.patch_cache = {}
            self.full_idx = full_img_idx
        else:
            cache_idx = int(idx) % self.patch_cache_size

        if self.patch_cache.get(cache_idx) is None:
            if self.split == Datasplit.TRAIN:
                gc.collect()
                self.populate_train_patch_cache()
            else:
                self.populate_test_patch_cache()
        cached_data = self.patch_cache[cache_idx]  # convoluted just for typing purposes
        assert cached_data is not None
        clean_patch, noisy_patch = cached_data
        self.patch_cache[cache_idx] = None

        clean_cond, clean_patch = self.get_conditioning(clean_patch) # type: ignore
        noisy_cond, noisy_patch = self.get_conditioning(noisy_patch) # type: ignore
        if clean_cond is not None:
            clean_cond = crop_valid(clean_cond, self.inflate_patches)
        return Batch(
            clean=clean_patch,
            corrupt=noisy_patch,
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
        if self.corruption is None:
            return torch.tensor(0.0)
        try:
            return self.corruption.noise_model.sigma
        except AttributeError:
            return torch.tensor(0.0)
