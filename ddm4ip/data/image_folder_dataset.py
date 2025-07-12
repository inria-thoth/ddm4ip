import abc
import io
import os
import re
from typing import AnyStr, ByteString, ClassVar, Dict, Generic, List, Literal, Tuple, TypeVar, Union
from typing_extensions import override
import dataclasses
import functools
from pathlib import Path
import warnings
import zipfile
import json

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2 as v2

from ddm4ip.data.utils import AddLocMapTransform
from ddm4ip.utils import distributed
from .base import Batch, DatasetType, Datasplit

T = TypeVar('T')


class AbstractImagePath(Generic[T], abc.ABC):
    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        pass

    @abc.abstractmethod
    def load_raw_img(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    @abc.abstractmethod
    def is_labels_path(self) -> bool:
        pass

    @abc.abstractmethod
    def read_file(self, mode) -> AnyStr:
        pass

    @staticmethod
    @abc.abstractmethod
    def load_all_paths(root: Path, **kwargs) -> List[T]:
        pass

    @abc.abstractmethod
    def is_image_path(self) -> bool:
        pass


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.pt', #'.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename: Union[str, Path]) -> bool:
    if isinstance(filename, Path) and not filename.is_file():
        return False
    return any(str(filename).endswith(extension) for extension in IMG_EXTENSIONS)


def file_ext(fname):
    return os.path.splitext(fname)[1].lower()


@dataclasses.dataclass
@functools.total_ordering
class ImagePath(AbstractImagePath['ImagePath']):
    root: Path
    name: str

    @override
    def __lt__(self, other: 'ImagePath'):
        return (self.root, self.name) < (other.root, other.name)

    @override
    def load_raw_img(self) -> torch.Tensor:
        # return: tensor, CHW, RGB, [0,255] (uint8)
        return torchvision.io.read_image(str((self.root / self.name).resolve()), mode=torchvision.io.ImageReadMode.RGB)

    @override
    def is_labels_path(self) -> bool:
        return self.name == "dataset.json"

    @override
    def read_file(self, mode="r"):
        with open(self.root / self.name, mode) as fh:
            return fh.read()

    @override
    def is_image_path(self) -> bool:
        return is_image_file(self.name)

    @override
    @staticmethod
    def load_all_paths(root: Path, *, filter: str | None = None, **kwargs) -> List['ImagePath']:
        if filter is not None:
            regex = re.compile(filter)
            filter_fn = lambda s: regex.match(s) is not None
        else:
            filter_fn = lambda s: True

        fnames = [str(f.relative_to(root)) for f in root.glob("**/*") if filter_fn(f.name)]
        fnames = sorted(fnames)
        return [ImagePath(root, fname) for fname in fnames]


@dataclasses.dataclass
@functools.total_ordering
class ZipImagePath(AbstractImagePath['ZipImagePath']):
    zip_cache: ClassVar[Dict[str, zipfile.ZipFile]] = dict()

    file: Union[str, Path]
    name: str

    @override
    def __lt__(self, other: 'ZipImagePath'):
        return (self.file, self.name) < (other.file, other.name)

    @override
    def load_raw_img(self) -> torch.Tensor:
        zfile = self.get_zipfile(self.file)
        with zfile.open(self.name, 'r') as f:
            imbuffer = f.read()
            if self.name.endswith(".pt"):
                bio_buf = io.BytesIO(imbuffer)
                return torch.load(bio_buf, weights_only=True)
            else:
                rwimbuffer = bytearray(imbuffer)
                torch_raw_img = torch.frombuffer(rwimbuffer, dtype=torch.uint8)
                try:
                    return torchvision.io.decode_image(torch_raw_img)
                except:
                    print(f"Error decoding {self.name}")
                    raise

    @override
    def read_file(self, mode: Literal["r"] = "r"):
        zfile = self.get_zipfile(self.file)
        with zfile.open(self.name, mode) as fh:
            return fh.read()

    @override
    def is_image_path(self) -> bool:
        return is_image_file(self.name)

    @override
    def is_labels_path(self) -> bool:
        return self.name == "dataset.json"

    @classmethod
    def get_zipfile(cls, path: Union[str, Path]) -> zipfile.ZipFile:
        if (file := cls.zip_cache.get(str(path))) is None:
            file = zipfile.ZipFile(str(path))
            cls.zip_cache[str(path)] = file
        return file

    @classmethod
    def clear_cache(cls):
        cls.zip_cache = dict()

    @override
    @staticmethod
    def load_all_paths(root: Path, **kwargs) -> List['ZipImagePath']:
        try:
            with zipfile.ZipFile(str(root)) as zfile:
                names = sorted(zfile.namelist())
                return [
                    ZipImagePath(root, f)
                    for f in names
                ]
        except zipfile.BadZipFile as e:
            raise zipfile.BadZipFile(f"Error when opening zip at {str(root)}: {e}") from e


class BaseImageFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        cache: bool,
        use_labels: bool,
        img_transform: v2.Transform,
        max_imgs: int | None = None,
        filter: str | None = None,
    ):
        self.path = Path(path)
        self.cache_images = cache
        self.img_cache = dict()  # {raw_idx: torch.Tensor, ...}
        self.img_transform = img_transform

        if os.path.isdir(self.path):
            all_paths = ImagePath.load_all_paths(self.path, filter=filter)
        elif file_ext(self.path) == '.zip':
            all_paths = ZipImagePath.load_all_paths(self.path)
        elif file_ext(self.path) == '.lmdb':
            all_paths = LMDBImagePath.load_all_paths(self.path)
        elif file_ext(self.path) in IMG_EXTENSIONS:
            all_paths = [ImagePath(self.path.parent, self.path.name)]
        else:
            raise IOError('Path must point to a directory or zip')

        img_paths = [p for p in all_paths if p.is_image_path()]
        if len(img_paths) == 0:
            raise IOError('No image files found in the specified path')
        self.raw_idx = torch.arange(len(img_paths), dtype=torch.int64)
        self.img_files = img_paths

        labels = None
        if use_labels:
            label_paths = [p for p in all_paths if p.is_labels_path()]
            assert len(label_paths) <= 1, f"Found {len(label_paths)} but expected at most 1."
            if len(label_paths) == 0:
                warnings.warn(f"No label file found for dataset at {self.path}")
                labels = None
            else:
                labels = self.load_labels_from_file(
                    label_file=label_paths[0],
                    img_fnames=[f.name for f in img_paths]
                )
        # Fix for when labels are not present
        if labels is None:
            labels = np.zeros([len(img_paths), 0], dtype=np.float32)
        self.labels = labels

        # Define label shape
        if self.labels.dtype == np.int64:
            self.label_shape = [int(np.max(self.labels)) + 1]
        else:
            self.label_shape = self.labels.shape[1:]

        # Apply `max_imgs`
        if max_imgs is not None:
            self.raw_idx = self.raw_idx[:max_imgs]
            self.labels = self.labels[self.raw_idx]
            self.img_files = [self.img_files[idx] for idx in self.raw_idx]

        # Checks on labels
        assert isinstance(self.labels, np.ndarray)
        assert self.labels.shape[0] == len(self.raw_idx)
        assert self.labels.dtype.name in {'float32', 'int64'}, f"Dtype is {self.labels.dtype=}"
        if self.labels.dtype == np.int64:
            assert self.labels.ndim == 1, f"Shape is {self.labels.shape=}"
            assert np.all(self.labels >= 0)

    def __len__(self):
        return len(self.raw_idx)

    def __getitem__(self, idx):
        raw_idx = self.raw_idx[idx]
        image = self.img_cache.get(raw_idx, None)
        if image is None:
            image = self.img_files[raw_idx].load_raw_img()
            if self.cache_images:
                self.img_cache[raw_idx] = image
        # assert list(image.shape) == self.raw_shape[1:]
        image = self.img_transform(image)
        return image, self.get_label(idx)

    def get_label(self, idx):
        label = self.labels[self.raw_idx[idx]]
        if label.dtype == np.int64:  # Convert to 1-hot
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def load_labels_from_file(self, label_file: AbstractImagePath, img_fnames: list[str | ByteString]):
        """
        dataset.json is in format of a list of tuples, one tuple per image.
        Each tuple is (relative img_path, label)
        """
        labels = json.loads(label_file.read_file("r"))['labels']
        labels = dict(labels)  # dict[img_path, label]
        img_labels = []
        for fname in img_fnames:
            if isinstance(fname, (bytes, bytearray)):
                fname = fname.decode()
            img_labels.append(
                labels[fname.replace('\\', '/')] # type: ignore
            )
        img_labels = np.array(img_labels)
        img_labels = img_labels.astype({1: np.int64, 2: np.float32}[img_labels.ndim])
        return img_labels

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]


class ZipFileDataset(torch.utils.data.Dataset[Batch], DatasetType):
    def __init__(
        self,
        path,
        degradation,
        split: Datasplit,
        dset_cfg,
        shuffle_clean: bool = True,
        generator=None,
    ):
        self.split = split
        self.shuffle_clean = shuffle_clean
        self.corruption = degradation
        self.cache = dset_cfg.get("cache", False)
        self.conditional = dset_cfg.get("cond", False)
        self.max_imgs = dset_cfg.get("max_imgs", None)
        self.x_flip = dset_cfg.get("x_flip", False) and self.split == Datasplit.TRAIN
        self.noisy_path = dset_cfg.get("noisy_path", None)
        self.space_conditioning = dset_cfg.get("space_conditioning", False)

        img_transform = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
        if self.space_conditioning:
            img_transform.append(AddLocMapTransform())
        if self.x_flip:
            img_transform.append(v2.RandomHorizontalFlip(p=0.5))

        self.full_dset = BaseImageFolderDataset(
            path=path,
            cache=self.cache,
            img_transform=v2.Compose(img_transform),
            use_labels=self.conditional,
            max_imgs=self.max_imgs,
        )
        self.noisy_dset = self.full_dset
        if self.noisy_path is not None:
            self.noisy_dset = BaseImageFolderDataset(
                path=self.noisy_path,
                cache=self.cache,
                img_transform=v2.Compose(img_transform),
                use_labels=self.conditional,
                max_imgs=self.max_imgs,
            )
            if len(self.full_dset) != len(self.noisy_dset):
                raise ValueError(
                    f"When noisy_path is specified, make sure noisy and clean datasets have "
                    f"the same length. Found {len(self.full_dset)=} and {len(self.noisy_dset)=}."
                )

        self.label_dim = self.full_dset.label_dim

        if shuffle_clean:
            self.clean_ids = torch.randperm(len(self.full_dset), generator=generator)
        else:
            self.clean_ids = torch.arange(len(self.full_dset))

        # Determine image channels and size
        sample_batch = self[0]
        sample_clean_cond = sample_batch.clean_conditioning
        sample_clean_img = sample_batch.clean
        assert sample_clean_img is not None
        self.clean_img_size = (sample_clean_img.shape[-3], sample_clean_img.shape[-2], sample_clean_img.shape[-1])
        self.clean_conditioning_channels = sample_clean_cond.shape[0] if sample_clean_cond is not None else 0

        sample_corrupt_cond = sample_batch.corrupt_conditioning
        sample_corrupt_img = sample_batch.corrupt
        assert sample_corrupt_img is not None
        self.corrupt_img_size = (sample_corrupt_img.shape[-3], sample_corrupt_img.shape[-2], sample_corrupt_img.shape[-1])
        self.corrupt_conditioning_channels = sample_corrupt_cond.shape[0] if sample_corrupt_cond is not None else 0

        distributed.print0(f"Loaded ImageFolder dataset at '{path}':")
        distributed.print0(f"Split:                         {self.split}")
        distributed.print0(f"Caching:                       {self.full_dset.cache_images}")
        distributed.print0(f"Number of images:              {len(self)}")
        distributed.print0(f"Label dimension:               {self.label_dim}")
        distributed.print0(f"Clean image size:              {self.clean_img_size}")
        distributed.print0(f"Noisy image size:              {self.corrupt_img_size}")
        distributed.print0(f"Space conditioning             {self.space_conditioning}")
        distributed.print0(f"Clean conditioning channels:   {self.clean_conditioning_channels}")
        distributed.print0(f"Noisy conditioning channels:   {self.corrupt_conditioning_channels}")
        distributed.print0()

    def get_conditioning(self, img: torch.Tensor | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        cond = None
        if self.space_conditioning and img is not None:
            cond = img[..., -2:, :, :]
            img = img[..., :-2, :, :]
        return cond, img

    def add_corruption(self, img, conditioning):
        if self.corruption is None:
            return img
        corrupt_img = self.corruption(
            img.unsqueeze(0), conditioning=conditioning,
        ).squeeze(0).clamp(0, 1)
        return corrupt_img

    def __getitem__(self, index):
        clean_idx = int(self.clean_ids[index % len(self)])
        clean_img, clean_tgt = self.full_dset[clean_idx]
        clean_conditioning, clean_img = self.get_conditioning(clean_img)

        noisy_idx = index % len(self)
        noisy_img, noisy_tgt = self.noisy_dset[noisy_idx]
        noisy_conditioning, noisy_img = self.get_conditioning(noisy_img)
        noisy_img = self.add_corruption(noisy_img, noisy_conditioning)

        if self.label_dim == 0:
            clean_tgt, noisy_tgt = None, None

        return Batch(
            clean=clean_img,
            corrupt=noisy_img,
            clean_label=clean_tgt, # type: ignore
            corrupt_label=noisy_tgt, # type: ignore
            noise_level=self.noise_level,
            clean_conditioning=clean_conditioning,
            corrupt_conditioning=noisy_conditioning,
        )

    @property
    def noise_level(self) -> torch.Tensor:
        if self.corruption is None:
            return torch.tensor(0.0)
        try:
            return self.corruption.noise_model.sigma
        except AttributeError:
            return torch.tensor(0.0)

    def __len__(self):
        return len(self.full_dset)
