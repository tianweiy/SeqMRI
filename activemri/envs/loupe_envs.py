import fastmri.data
import pathlib
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets
from torchvision import transforms as TF
from torch.utils.data import DataLoader, Subset, random_split

import activemri.data.singlecoil_knee_data as scknee_data
import activemri.data.transforms
import activemri.envs.masks
import activemri.envs.util
import fastmri
from activemri.data.real_knee_data import RealKneeData
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)
from activemri.envs.envs import ActiveMRIEnv, _env_collate_fn

def transform(
    kspace: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    attrs: List[Dict[str, Any]],
    fname: List[str],
    slice_id: List[int],
) -> Tuple:
    label = attrs
    return torch.from_numpy(kspace), mask, torch.from_numpy(target), label

class LOUPEDataEnv():
    def __init__(self, options):
        self._data_location = options.data_path
        self.options = options

    def _setup_data_handlers(self):
        train_data, val_data, test_data = self._create_datasets()

        display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 16)]

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.options.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.options.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.options.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=True
        )
        display_loader = DataLoader(
            dataset=display_data,
            batch_size=16,
            num_workers=8,
            shuffle=False,
            pin_memory=True
        )
        return train_loader, val_loader, test_loader, display_loader


class LOUPEActiveMRIEnv(LOUPEDataEnv):
    def __init__(self, options):
        super().__init__(options)

    def _create_datasets(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_and_test_path = root_path / "knee_singlecoil_val"

        train_data = scknee_data.MICCAI2020Data(
            train_path,
            ActiveMRIEnv._void_transform,
            num_cols=self.options.resolution[1],
        )
        val_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="val",
            num_cols=self.options.resolution[1],
        )
        test_data = scknee_data.MICCAI2020Data(
            val_and_test_path,
            ActiveMRIEnv._void_transform,
            custom_split="test",
            num_cols=self.options.resolution[1],
        )
        return train_data, val_data, test_data

class LOUPERealKspaceEnv(LOUPEDataEnv):
    def __init__(self, options):
        super().__init__(options)

    def _create_datasets(self):
        root_path = pathlib.Path(self._data_location)
        train_path = root_path / "knee_singlecoil_train"
        val_path = root_path / "knee_singlecoil_val"
        test_path = root_path / "knee_singlecoil_val"

        train_data = RealKneeData(
            train_path,
            self.options.resolution,
            ActiveMRIEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            random_rotate=self.options.random_rotate 
        )
        val_data = RealKneeData(
            val_path,
            self.options.resolution,
            ActiveMRIEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            custom_split='val',
            random_rotate=self.options.random_rotate 
        )
        test_data = RealKneeData(
            test_path,
            self.options.resolution,
            ActiveMRIEnv._void_transform,
            self.options.noise_type,
            self.options.noise_level,
            custom_split='test',
            random_rotate=self.options.random_rotate 
        )

        return train_data, val_data, test_data

class LOUPEFashionMNISTEnv(LOUPEDataEnv):
    def __init__(self, options):
        super().__init__(options)

    def get_same_index(target, label):
        label_indices = []

        for i in range(len(target)):
            if target[i] in label:
                label_indices.append(i)

        return label_indices

    def _create_datasets(self):
        train_val_set = FashionMNISTData(args=self.options, root=self._data_location, transform=transform,
            noise_type=self.options.noise_type, noise_level=self.options.noise_level, custom_split='train',
            label_range=self.options.label_range)
        test_set = FashionMNISTData(args=self.options, root=self._data_location, transform=transform,
            noise_type=self.options.noise_type, noise_level=self.options.noise_level, custom_split='test',
            label_range=self.options.label_range)


        val_set_size = int(len(train_val_set)/4)
        train_set_size = len(train_val_set) - val_set_size
        train_set, val_set = random_split(train_val_set, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(42))


        if self.options.sample_rate < 1:
            # create a random subset
            train_set = random_split(train_set, [int(self.options.sample_rate * len(train_set)),
                                    len(train_set)-int(self.options.sample_rate * len(train_set))], generator=torch.Generator().manual_seed(42))[0]

            val_set = random_split(val_set, [int(self.options.sample_rate * len(val_set)),
                                    len(val_set)-int(self.options.sample_rate * len(val_set))], generator=torch.Generator().manual_seed(42))[0]

        return train_set, val_set, test_set

class LOUPESyntheticEnv(LOUPEDataEnv):
    def __init__(self, options):
        super().__init__(options)

    def _create_datasets(self):
        train_set = SyntheticData(root=self._data_location, split='train')
        val_set = SyntheticData(root=self._data_location, split='val')
        test_set = SyntheticData(root=self._data_location, split='test')

        return train_set, val_set, test_set
