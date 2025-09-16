import os
import abc
from typing import Any, Tuple, ClassVar, Sequence
from typing_extensions import override
from pathlib import Path
import functools
import zipfile
import dataclasses

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms.v2 as v2
import deepinv

from ddm4ip.data.base import Batch, DatasetType, Datasplit
from ddm4ip.data.utils import (
    AddLocMapTransform,
    AddRandomLocMapTransform,
    GetAllPatchesTransform,
    ReplaceRandomLocMapTransform,
    get_dim_blocks,
)
from ddm4ip.utils.torch_utils import crop_valid
from ddm4ip.utils import distributed



class AbstractImagePath(abc.ABC):
    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def load_raw_img(self) -> Image.Image:
        pass

    @staticmethod
    @abc.abstractmethod
    def load_all_paths(root: Path, **kwargs) -> Sequence['AbstractImagePath']:
        pass


@dataclasses.dataclass
@functools.total_ordering
class ImagePath(AbstractImagePath):
    root: Path
    name: str

    @override
    def __lt__(self, other: 'ImagePath'):
        return (self.root, self.name) < (other.root, other.name)

    @override
    def load_raw_img(self) -> Image.Image:
        # return: tensor, CHW, RGB, [0,255] (uint8)
        with open(self.root / self.name, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    @override
    @staticmethod
    def load_all_paths(root: Path, **kwargs):
        fnames = [
            str(f.relative_to(root))
            for f in Path(root).rglob("*")
            if f.suffix in torchvision.datasets.folder.IMG_EXTENSIONS
        ]
        fnames = sorted(fnames)
        return [ImagePath(root, fname) for fname in fnames]

    def __hash__(self) -> int:
        return hash((self.root, self.name))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ImagePath):
            return self.root == value.root and self.name == value.name
        return False


@dataclasses.dataclass
@functools.total_ordering
class ZipImagePath(AbstractImagePath):
    zip_cache: ClassVar[dict[str, zipfile.ZipFile]] = dict()

    file: str | Path
    name: str

    @override
    def __lt__(self, other: 'ZipImagePath'):
        return (self.file, self.name) < (other.file, other.name)

    @override
    def load_raw_img(self) -> Image.Image:
        zfile = self.get_zipfile(self.file)
        with zfile.open(self.name, 'r') as f:
            img = Image.open(f)
            return img.convert("RGB")

    @classmethod
    def get_zipfile(cls, path: str | Path) -> zipfile.ZipFile:
        if (file := cls.zip_cache.get(str(path))) is None:
            file = zipfile.ZipFile(str(path))
            cls.zip_cache[str(path)] = file
        return file

    @classmethod
    def clear_cache(cls):
        cls.zip_cache = dict()

    @override
    @staticmethod
    def load_all_paths(root: Path, **kwargs):
        try:
            with zipfile.ZipFile(str(root)) as zfile:
                names = sorted(
                    filter(
                        lambda n: os.path.splitext(n)[1] in torchvision.datasets.folder.IMG_EXTENSIONS,
                        zfile.namelist()
                    )
                )
                return [ZipImagePath(root, f) for f in names]
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(f"Error when opening zip at {str(root)}: {e}") from e

    def __hash__(self) -> int:
        return hash((self.file, self.name))

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ZipImagePath):
            return self.file == value.file and self.name == value.name
        return False


def load_all_paths(root: Path):
    # TODO: Make sure this function is okay for all datasets we have
    if root.suffix == ".zip" and root.is_file():
        return ZipImagePath.load_all_paths(root)
    elif root.is_dir():
        # Load from fs
        all_files = ImagePath.load_all_paths(root)
        # Load from zips
        for zip_file in root.rglob("*.zip"):
            all_files.extend(ZipImagePath.load_all_paths(zip_file)) # pyright: ignore[reportArgumentType]
        return all_files
    else:
        raise ValueError(f"{root} not recognized as image dataset root.")


def inflate_patch_size(patch_size: int, filter_size: int) -> int:
    pad_before, pad_after = 0, 0
    if filter_size > 0:
        pad_before = filter_size // 2
        pad_after = filter_size // 2 - (filter_size - 1) % 2
    return patch_size + pad_before + pad_after


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


class ImageFolderDataset(torch.utils.data.Dataset[torch.Tensor]):
    """Base dataset loading images from a folder or zip file"""
    def __init__(self, base_path: Path | str, transforms: v2.Transform | None, shuffle: bool):
        self.transforms = transforms if transforms is not None else v2.Identity()
        if isinstance(base_path, str):
            base_path = os.path.expanduser(base_path)
        self.root = Path(base_path)
        self.files = load_all_paths(self.root)
        if shuffle:
            self.files = [self.files[i] for i in torch.randperm(len(self.files))]

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.files[index]
        return self.transforms(path.load_raw_img())

    def __len__(self) -> int:
        return len(self.files)


class BaseDataset(torch.utils.data.Dataset[Batch], DatasetType, abc.ABC):
    def __init__(self, split: Datasplit, dset_cfg: dict[str, Any], shuffle_clean: bool = True):
        self.split = split
        self.dset_cfg = dset_cfg
        self.shuffle_clean = shuffle_clean
        self.patch_size: int | None = dset_cfg.get("patch_size", None)
        self.inflate_patches: int = dset_cfg.get("inflate_patches", 0)
        self.label_dim = 0  # NOTE: always fixed, should be removed.

        # Parse conditioning parameters
        self.space_conditioning: bool = dset_cfg.get("space_conditioning", False)
        self.random_space_conditioning: bool = dset_cfg.get("random_space_conditioning", False)
        if self.space_conditioning and self.random_space_conditioning:
            raise ValueError("Space conditioning and random space conditioning cannot both be active.")
        self.randomize_conditioning: bool | float = float(dset_cfg.get("randomize_conditioning", 0.0))
        if self.randomize_conditioning > 0 and not self.space_conditioning:
            raise ValueError("Cannot randomize conditioning unless `space_conditioning` is True.")

        self.need_clean: bool = dset_cfg.get("need_clean", True)
        self.need_noisy: bool = dset_cfg.get("need_noisy", True)
        assert self.need_clean or self.need_noisy

        self.num_patches_per_image = 1
        if self.patch_size is not None and self.split == Datasplit.TRAIN:
            self.num_patches_per_image = dset_cfg.get("num_patches_per_image", 1)

        self.init_base_dset()
        self.dset_length = self.init_length()

        # Set-up necessary attributes by running pipeline
        batch = self[0]
        self.clean_img_size = (0, 0, 0)
        if batch.clean is not None:
            self.clean_img_size = (batch.clean.shape[-3], batch.clean.shape[-2], batch.clean.shape[-1])
        self.clean_conditioning_channels = 0
        if batch.clean_conditioning is not None:
            self.clean_conditioning_channels = batch.clean_conditioning.shape[-3]
        self.corrupt_img_size = (0, 0, 0)
        if batch.corrupt is not None:
            self.corrupt_img_size = (batch.corrupt.shape[-3], batch.corrupt.shape[-2], batch.corrupt.shape[-1])
        self.corrupt_conditioning_channels = 0
        if batch.corrupt_conditioning is not None:
            self.corrupt_conditioning_channels = batch.corrupt_conditioning.shape[-3]

    @abc.abstractmethod
    def init_base_dset(self):
        pass

    @abc.abstractmethod
    def init_length(self) -> int:
        pass

    def get_num_patches(self, img: torch.Tensor):
        if self.patch_size is None:
            return 1
        act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
        num_img_patches_h = get_dim_blocks(
            img.shape[-2], act_patch_size, stride=self.patch_size
        )
        num_img_patches_w = get_dim_blocks(
            img.shape[-1], act_patch_size, stride=self.patch_size
        )
        return num_img_patches_h * num_img_patches_w

    def split_img_cond(self, full: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        img, cond = None, None
        if (self.space_conditioning or self.random_space_conditioning is not False) and full is not None:
            cond = full[..., -2:, :, :]
            img = full[..., :-2, :, :]
        elif full is not None:
            img = full
        return img, cond

    def add_cond(self, img: torch.Tensor, conditioning: torch.Tensor | None) -> torch.Tensor:
        if conditioning is None:
            return img
        if img.shape[-1] == conditioning.shape[-1] and img.shape[-2] == conditioning.shape[-2]:
            return torch.cat((img, conditioning), dim=-3)
        elif self.inflate_patches > 0:
            return torch.cat((img, crop_valid(conditioning, self.inflate_patches)), dim=-3)
        raise RuntimeError(f"Cannot add conditioning of shape {conditioning.shape} to image of shape {img.shape}")

    def get_train_transform(self, img_size: tuple[int, int]) -> v2.Transform:
        trsf = []
        if self.patch_size is not None:
            act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
            trsf.append(v2.RandomCrop(act_patch_size, pad_if_needed=True))
        if self.randomize_conditioning > 0:
            trsf.append(ReplaceRandomLocMapTransform(self.randomize_conditioning, img_size))
        if self.random_space_conditioning is not False:
            trsf.append(AddRandomLocMapTransform(img_size))
        return v2.Compose(trsf)

    @abc.abstractmethod
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass

    def get_test_transform(self, img_size: tuple[int, int]) -> v2.Transform:
        trsf = []
        if self.patch_size is not None:
            act_patch_size = inflate_patch_size(self.patch_size, self.inflate_patches)
            trsf.append(GetAllPatchesTransform(act_patch_size, stride=self.patch_size))
        else:
            trsf.append(v2.Identity())
        return v2.Compose(trsf)

    @abc.abstractmethod
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass

    def __getitem__(self, index: int) -> Batch:
        if self.split == Datasplit.TEST:
            clean_full, noisy_full = self.getitem_test(index)
        else:
            clean_full, noisy_full = self.getitem_train(index)

        clean_patch, clean_cond = self.split_img_cond(clean_full)
        noisy_patch, noisy_cond = self.split_img_cond(noisy_full)
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

    @property
    @abc.abstractmethod
    def noise_level(self) -> torch.Tensor | None:
        pass

    def __len__(self):
        return self.dset_length


class KnownCorruptionDataset(BaseDataset):
    def __init__(
        self,
        path: str | Path,
        corruption: deepinv.physics.Physics,
        split: Datasplit,
        dset_cfg: dict[str, Any],
        shuffle_clean: bool = True,
    ):
        self.path = Path(path)
        self.corruption = corruption
        # Will be initialized
        self.dset = None
        self.clean_cache, self.noisy_cache = None, None
        self.clean_full_ids = None
        self.tst_n_patches = None
        super().__init__(split, dset_cfg, shuffle_clean)
        assert self.dset is not None

        distributed.print0(f"Loaded KnownCorruptionDataset dataset at '{path}':")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Inflate patches (k-size):       {self.inflate_patches}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Space conditioning:             {'normal' if self.space_conditioning else 'random' if self.random_space_conditioning else 'none'}")
        distributed.print0(f"Random replace conditioning p:  {self.randomize_conditioning}")
        distributed.print0(f"Number of base images:          {len(self.dset)}")
        if self.split == Datasplit.TEST:
            distributed.print0(f"Dataset length:                 {self.dset_length}")
        else:
            distributed.print0(f"Number patches per image:       {self.num_patches_per_image}")
        distributed.print0()

    @override
    def init_base_dset(self):
        img_trsf = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.space_conditioning:
            img_trsf.append(AddLocMapTransform())
        self.dset = ImageFolderDataset(
            base_path=self.path, transforms=v2.Compose(img_trsf), shuffle=False
        )

        # Initialize separate indices into the full dataset for clean
        if self.split == Datasplit.TRAIN and self.need_clean:
            if self.shuffle_clean:
                self.clean_full_ids = torch.randperm(len(self.dset)).tolist()
            else:
                self.clean_full_ids = torch.arange(len(self.dset)).tolist()

    @override
    def init_length(self):
        assert self.dset is not None
        if self.split == Datasplit.TRAIN:
            return self.num_patches_per_image * len(self.dset)
        else:
            self.tst_n_patches = [self.get_num_patches(full_img) for full_img in self.dset]
            return sum(self.tst_n_patches)

    def corrupt_patch(self, full_patch: torch.Tensor) -> torch.Tensor:
        patch, cond = self.split_img_cond(full_patch)
        assert patch is not None
        squeeze_out = False
        if patch.dim() == 3:
            squeeze_out = True
            patch = patch[None, ...]
        if cond is not None and cond.dim() == 3:
            cond = cond[None, ...]
        corrupted = self.corruption(patch, conditioning=cond)
        full_patch = self.add_cond(corrupted, cond)
        if squeeze_out:
            return full_patch.squeeze(0)
        return full_patch

    @override
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.dset is not None
        clean_full, noisy_full = None, None
        if self.need_clean:  # get clean patch with conditioning
            assert self.clean_full_ids is not None
            if self.clean_cache is None or len(self.clean_cache) == 0:
                self.clean_cache = []
                img = self.dset[self.clean_full_ids[index // self.num_patches_per_image]]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    self.clean_cache.append(trsf(img))
            assert isinstance(self.clean_cache, list)
            clean_full = self.clean_cache.pop()
        if self.need_noisy:  # get noisy patch with conditioning
            if self.noisy_cache is None or len(self.noisy_cache) == 0:
                self.noisy_cache = []
                img = self.dset[index // self.num_patches_per_image]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    # NOTE: Patch corruption happens here.
                    # TODO: Test if it's faster to corrupt in __getitem__
                    self.noisy_cache.append(self.corrupt_patch(trsf(img)))
            assert isinstance(self.noisy_cache, list)
            noisy_full = self.noisy_cache.pop()
        return clean_full, noisy_full

    @override
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.tst_n_patches is not None
        assert self.dset is not None
        clean_full, noisy_full = None, None
        img_idx, patch_idx = search_idx_in_list(self.tst_n_patches, idx=index)
        # We fetch image only once and always put in the clean cache
        if self.clean_cache is None or self.clean_cache[0] != img_idx:
            self.clean_cache = (img_idx, [])
            img = self.dset[img_idx]
            trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
            for patch in trsf(img).unbind(0):
                self.clean_cache[1].append(patch)

        if self.need_clean:
            clean_full = self.clean_cache[1][patch_idx]
        if self.need_noisy:
            noisy_full = self.corrupt_patch(self.clean_cache[1][patch_idx])

        return clean_full, noisy_full

    @property
    def noise_level(self) -> torch.Tensor | None:
        try:
            return self.corruption.noise_model.sigma
        except AttributeError:
            return None


class UnknownCorruptionDataset(BaseDataset):
    def __init__(
        self,
        clean_path: str | Path,
        noisy_path: str | Path,
        split: Datasplit,
        dset_cfg: dict[str, Any],
        shuffle_clean: bool = True,
    ):
        self.clean_path = Path(clean_path)
        self.noisy_path = Path(noisy_path)
        self.corruption = None
        self.clean_dset, self.noisy_dset = None, None
        self.clean_cache, self.noisy_cache = None, None
        self.tst_n_patches = None
        dset_cfg["inflate_patches"] = 0  # force to be 0
        super().__init__(split, dset_cfg, shuffle_clean)

        distributed.print0(f"Loaded UnknownCorruptionDataset dataset at '{self.clean_path}' and '{self.noisy_path}':")
        distributed.print0(f"Split:                          {self.split}")
        distributed.print0(f"Clean patch size:               {self.clean_img_size}")
        distributed.print0(f"Noisy patch size:               {self.corrupt_img_size}")
        distributed.print0(f"Clean conditioning channels:    {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:    {self.corrupt_conditioning_channels}")
        distributed.print0(f"Space conditioning:             {'normal' if self.space_conditioning else 'random' if self.random_space_conditioning else 'none'}")
        distributed.print0(f"Random replace conditioning p:  {self.randomize_conditioning}")
        distributed.print0(f"Number of base clean images:    {len(self.clean_dset) if self.clean_dset is not None else 0}")
        distributed.print0(f"Number of base corrupt images:  {len(self.noisy_dset) if self.noisy_dset is not None else 0}")
        if self.split == Datasplit.TEST:
            distributed.print0(f"Dataset length:                 {self.dset_length}")
        else:
            distributed.print0(f"Number patches per image:       {self.num_patches_per_image}")
        distributed.print0()

    @override
    def init_base_dset(self):
        img_trsf = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        if self.space_conditioning:
            img_trsf.append(AddLocMapTransform())
        if self.need_clean:
            self.clean_dset = ImageFolderDataset(
                base_path=self.clean_path, transforms=v2.Compose(img_trsf), shuffle=self.shuffle_clean
            )
        if self.need_noisy:
            self.noisy_dset = ImageFolderDataset(
                base_path=self.noisy_path, transforms=v2.Compose(img_trsf), shuffle=False
            )
        if (self.split == Datasplit.TEST and
            self.clean_dset is not None and self.noisy_dset is not None and
            len(self.clean_dset) != len(self.noisy_dset)
            ):
            raise RuntimeError("Test datasets cannot have different lengths as they are paired.")

    @override
    def init_length(self):
        if self.split == Datasplit.TRAIN:
            return self.num_patches_per_image * max(
                len(self.clean_dset) if self.clean_dset is not None else 0,
                len(self.noisy_dset) if self.noisy_dset is not None else 0
            )
        else:
            tst_dset = self.clean_dset or self.noisy_dset
            assert tst_dset is not None
            self.tst_n_patches = [self.get_num_patches(full_img) for full_img in tst_dset]
            return sum(self.tst_n_patches)

    @override
    def getitem_train(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        clean_full, noisy_full = None, None
        if self.need_clean:  # get clean patch with conditioning
            assert self.clean_dset is not None
            if self.clean_cache is None or len(self.clean_cache) == 0:
                self.clean_cache = []
                img = self.clean_dset[(index // self.num_patches_per_image) % len(self.clean_dset)]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    self.clean_cache.append(trsf(img))
            assert isinstance(self.clean_cache, list)
            clean_full = self.clean_cache.pop()
        if self.need_noisy:  # get noisy patch with conditioning
            assert self.noisy_dset is not None
            if self.noisy_cache is None or len(self.noisy_cache) == 0:
                self.noisy_cache = []
                img = self.noisy_dset[(index // self.num_patches_per_image) % len(self.noisy_dset)]
                trsf = self.get_train_transform((img.shape[-2], img.shape[-1]))
                for _ in range(self.num_patches_per_image):
                    self.noisy_cache.append(trsf(img))
            assert isinstance(self.noisy_cache, list)
            noisy_full = self.noisy_cache.pop()
        return clean_full, noisy_full

    @override
    def getitem_test(self, index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert self.tst_n_patches is not None
        clean_full, noisy_full = None, None
        img_idx, patch_idx = search_idx_in_list(self.tst_n_patches, idx=index)
        if self.need_clean:
            assert self.clean_dset is not None
            if self.clean_cache is None or self.clean_cache[0] != img_idx:
                self.clean_cache = (img_idx, [])
                img = self.clean_dset[img_idx]
                trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
                for patch in trsf(img).unbind(0):
                    self.clean_cache[1].append(patch)
            clean_full = self.clean_cache[1][patch_idx]
        if self.need_noisy:
            assert self.noisy_dset is not None
            if self.noisy_cache is None or self.noisy_cache[0] != img_idx:
                self.noisy_cache = (img_idx, [])
                img = self.noisy_dset[img_idx]
                trsf = self.get_test_transform((img.shape[-2], img.shape[-1]))
                for patch in trsf(img).unbind(0):
                    self.noisy_cache[1].append(patch)
            noisy_full = self.noisy_cache[1][patch_idx]
        return clean_full, noisy_full

    @property
    def noise_level(self) -> torch.Tensor | None:
        """Noise must be learned from data."""
        return None
