"""Colorised MNIST datamodule."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union

import ethicml.vision as emvi
import torch
from ethicml import implements
from ethicml.vision import LdColorizer, LdTransformation
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from fair_bolts.datamodules.vision_datamodule import BaseDm
from fair_bolts.datasets.ethicml_datasets import Batch


class CmnistDataModule(BaseDm):
    """Lightning Data Module for CMNIST."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        colours: Optional[List[int]] = None,
        correlation: float = 1.0,
        label_map: Optional[Dict[str, int]] = None,
        num_colours: int = 10,
        scale: float = 0.2,
        num_workers: int = 0,
        shrink_pcnt: float = 1.0,
        val_split: float = 0.2,
    ):
        self.num_classes = max(label_map.values()) + 1 if label_map is not None else 10
        self.num_colours = num_colours
        y_dim = 1 if self.num_classes == 2 else self.num_classes
        s_dim = 1 if self.num_colours == 2 else self.num_colours
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            shrink_pcnt=shrink_pcnt,
            y_dim=y_dim,
            s_dim=s_dim,
        )
        self.dims = (3, 32, 32)

        self.correlation = correlation
        self.scale = scale
        self.label_map = label_map if label_map is not None else {i: i for i in range(10)}
        self.colours = colours

    def _filter(self, dataset: MNIST) -> None:
        final_mask = torch.zeros_like(dataset.targets).bool()
        for old_label, new_label in self.label_map.items():
            mask = dataset.targets == int(old_label)
            dataset.targets[mask] = new_label
            final_mask |= mask
        dataset.data = dataset.data[final_mask]
        dataset.targets = dataset.targets[final_mask]

    @implements(LightningDataModule)
    def prepare_data(self) -> None:
        _ = MNIST(root=self.data_dir, download=True, train=True)
        _ = MNIST(root=self.data_dir, download=True, train=False)

    @implements(LightningDataModule)
    def setup(self, stage: Optional[str] = None) -> None:
        base_aug = [transforms.ToTensor()]
        base_aug.insert(0, transforms.Pad(2))
        train_data = MNIST(root=self.data_dir, download=False, train=True)
        test_data = MNIST(root=self.data_dir, download=False, train=False)

        self._filter(train_data)
        self._filter(test_data)

        train_len = int(len(train_data) * self.shrink_pcnt)

        val_len = round(self.val_pcnt * train_len)
        train_len -= val_len
        val_data, train_data, _ = random_split(
            train_data, lengths=(val_len, train_len, len(train_data) - train_len - val_len)
        )

        test_len = int(len(test_data) * self.shrink_pcnt)
        test_data, _ = random_split(test_data, lengths=(test_len, len(test_data) - test_len))

        colorizer = LdColorizer(
            scale=self.scale,
            background=False,
            black=True,
            binarize=False,
            greyscale=False,
            color_indices=self.colours,
        )

        self.val_data = LdAugmentedDataset(
            val_data,
            ld_augmentations=colorizer,
            num_classes=self.num_classes,
            num_colours=self.num_colours,
            li_augmentation=True,
            base_augmentations=base_aug,
        )
        self.train_data = LdAugmentedDataset(
            train_data,
            ld_augmentations=colorizer,
            num_classes=self.num_classes,
            num_colours=self.num_colours,
            li_augmentation=False,
            base_augmentations=base_aug,
            correlation=self.correlation,
        )
        self.test_data = LdAugmentedDataset(
            test_data,
            ld_augmentations=colorizer,
            num_classes=self.num_classes,
            num_colours=self.num_colours,
            li_augmentation=True,
            base_augmentations=base_aug,
        )


class LdAugmentedDataset(Dataset):
    """Do Color Augmentations."""

    def __init__(
        self,
        source_dataset: Dataset,
        ld_augmentations: LdTransformation,
        num_classes: int,
        num_colours: int,
        li_augmentation: bool = False,
        base_augmentations: Optional[List] = None,
        correlation: float = 1.0,
    ) -> None:
        self.source_dataset = self._validate_dataset(source_dataset)
        if not 0 <= correlation <= 1:
            raise ValueError("Label-augmentation correlation must be between 0 and 1.")

        self.num_classes = num_classes
        self.num_colours = num_colours
        self.inds: Optional[List[Tensor]] = None

        if not isinstance(ld_augmentations, (list, tuple)):
            ld_augmentations = [ld_augmentations]
        self.ld_augmentations = ld_augmentations

        if base_augmentations is not None:
            base_augmentations = transforms.Compose(base_augmentations)
            emvi.set_transform(self.source_dataset, base_augmentations)
        self.base_augmentations = base_augmentations

        self.li_augmentation = li_augmentation
        self.correlation = correlation

        self.dataset = self._validate_dataset(source_dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _validate_dataset(dataset: Dataset | DataLoader) -> Dataset:
        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset
        elif not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be a Dataset or Dataloader object.")

        return dataset

    def subsample(self, pcnt: float = 1.0) -> None:
        """Subsample."""
        if not 0 <= pcnt <= 1.0:
            raise ValueError(f"{pcnt} should be in the range (0, 1]")
        num_samples = int(pcnt * len(self.source_dataset))
        inds = list(RandomSampler(self.source_dataset, num_samples=num_samples, replacement=False))
        self.inds = inds
        subset = self._sample_from_inds(inds)
        self.dataset = subset

    def _sample_from_inds(self, inds: Tensor) -> Dataset:
        return Subset(self.source_dataset, inds)

    @staticmethod
    def _validate_data(*args: Union[Tensor, int]) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                dtype = torch.long if isinstance(arg, int) else torch.float32  # type: ignore[comparison-overlap]
                arg = torch.tensor(arg, dtype=dtype)
            if arg.dim() == 0:
                arg = arg.view(-1)
            yield arg

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return self._subroutine(self.dataset.__getitem__(index))

    def _augment(self, x: Tensor, label: Tensor) -> Tensor:
        for aug in self.ld_augmentations:
            x = aug(x, label)

        return x

    def _subroutine(self, data: Tuple[Tensor, Tensor]) -> Batch:

        x, y = data
        s = y % self.num_colours
        x, s, y = self._validate_data(x, s, y)

        if self.li_augmentation:
            s = torch.randint_like(s, low=0, high=self.num_colours)

        if self.correlation < 1:
            flip_prob = torch.rand(s.shape)  # type: ignore[attr-defined]
            if flip_prob > self.correlation:
                s = torch.randint_like(s, low=0, high=self.num_colours)  # type: ignore[index]

        x = self._augment(x, s)

        if x.dim() == 4 and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.squeeze(0)
        s = s.squeeze()  # type: ignore[attr-defined]

        return Batch(x=x, s=s, y=y)