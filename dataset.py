"""
Pytorch Lightning Datamodule and Dataset implementation.
"""

import lightning as L
import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader


class NomadicLocalizationDataset(Dataset):
    def __init__(self, data_dir, sample_ids, labels, augment_method=None):
        self.labels = labels[:, :2].astype(np.float32)
        self.sample_ids = sample_ids
        self.sample_ids_len = len(self.sample_ids)

        self.samples = []
        for sample_id in sample_ids:
            sample = np.load(os.path.join(data_dir, "samples", f"channel_measurement_{str(sample_id).zfill(6)}.npy"))
            sample_real = sample.real
            sample_imag = sample.imag
            sample = np.stack([sample_real, sample_imag], axis=0)
            self.samples.append(sample)

        self.augment_method = augment_method

        # Load all samples at initialization
        if self.augment_method:
            nr_augmented_samples = int(len(self.sample_ids) * 0.5)
            self.idc_to_augment = np.random.choice(range(len(self.sample_ids)), nr_augmented_samples, replace=False)
            if self.augment_method in ["vanilla", "random attenuation"]:
                blocked_antennas = [np.random.choice(np.arange(64), nr_blocked_antennas, replace=False) for
                                         nr_blocked_antennas in [np.random.randint(1, 64) for _ in
                                                                 range(nr_augmented_samples)]]
                # Pre-compute augmented samples
                for sample_idx, antenna_idc in zip(self.idc_to_augment, blocked_antennas):
                    augmented_sample = self.samples[sample_idx].copy()
                    if self.augment_method == "vanilla":
                        augmented_sample[:, antenna_idc, :] = 0
                    else:
                        augmented_sample[0, antenna_idc, :] *= np.random.uniform(0.01, 0.3,
                                                                                 size=(antenna_idc.shape[0], 1))
                        augmented_sample[1, antenna_idc, :] *= np.random.uniform(0.01, 0.3,
                                                                                 size=(antenna_idc.shape[0], 1))
                    self.samples.append(augmented_sample)
            else:
                raise ValueError("Invalid augmentation method.")

        self.samples = np.array(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if idx >= self.sample_ids_len:
            true_idx = idx - self.sample_ids_len
            idx = self.idc_to_augment[true_idx]
        sample_id = self.sample_ids[idx]
        label_idx = int(str(sample_id).zfill(6)[2])
        label = self.labels[label_idx]
        return torch.from_numpy(sample).to(dtype=torch.float32), torch.from_numpy(label).to(dtype=torch.float32)


class NomadicLocalizationDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size: int = 128, num_workers: int = 4, num_antennas: int = 64,
                 num_subcarriers: int = 100, num_users = 4, num_samples = 240, num_scenarios = 7, mode="test_only",
                 sample_ids=None, augment_method=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers

        self.num_users = num_users
        self.num_samples = num_samples
        self.num_scenarios = num_scenarios

        assert mode in ["test_only", "train_static", "train_nomadic"], \
            "Unknown mode. Choose from 'test_only', 'train_static' or 'train_nomadic'."
        self.mode = mode
        self.sample_ids = sample_ids

        if augment_method and mode == "test_only":
            raise ValueError("Cannot augment test data.")
        if augment_method and mode == "train_nomadic":
            raise ValueError("Cannot augment nomadic data.")
        self.augment_method = augment_method

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._already_setup = False

    def setup(self, stage=None):
        if self._already_setup:
            return
        self._already_setup = True

        user_positions = np.load(os.path.join(self.data_dir, "user_positions.npy"))
        user_range = range(self.num_users) if isinstance(self.num_users, int) else self.num_users

        np.random.seed(42)

        # Test on all scenarios
        if self.mode == "test_only":
            if self.sample_ids is None:
                self.sample_ids = [t * 10000 + u * 1000 + s for t in range(self.num_scenarios) for u in user_range for s
                                   in range(self.num_samples)]
            self.test_dataset = NomadicLocalizationDataset(self.data_dir, self.sample_ids, user_positions)
        # Finetune and test on static reference scenario
        elif self.mode == "train_static":
            num_train = int(self.num_samples * 0.7)
            num_val = int(self.num_samples * 0.15)
            train_IDs, val_IDs, test_IDs = [], [], []
            for user in user_range:
                IDs_per_user = [60000 + user * 1000 + s for s in range(self.num_samples)]
                np.random.shuffle(IDs_per_user)
                train_IDs.append(IDs_per_user[:num_train])
                val_IDs.append(IDs_per_user[num_train:num_train + num_val])
                test_IDs.append(IDs_per_user[num_train + num_val:])
            train_IDs = np.array(train_IDs).flatten()
            val_IDs = np.array(val_IDs).flatten()
            test_IDs = np.array(test_IDs).flatten()
            self.train_dataset = NomadicLocalizationDataset(self.data_dir, train_IDs, user_positions,
                                                            augment_method=self.augment_method)
            self.val_dataset = NomadicLocalizationDataset(self.data_dir, val_IDs, user_positions)
            self.test_dataset = NomadicLocalizationDataset(self.data_dir, test_IDs, user_positions)

        elif self.mode == "train_nomadic":
            num_train = int(self.num_samples * 0.7)
            num_val = int(self.num_samples * 0.15)
            train_IDs, val_IDs, test_IDs = [], [], []
            for user in user_range:
                for scenario in range(self.num_scenarios - 1):
                    IDs_per_user_per_scenario = [scenario * 10000 + user * 1000 + s for s in range(self.num_samples)]
                    np.random.shuffle(IDs_per_user_per_scenario)
                    train_IDs.append(IDs_per_user_per_scenario[:num_train])
                    val_IDs.append(IDs_per_user_per_scenario[num_train:num_train + num_val])
                    test_IDs.append(IDs_per_user_per_scenario[num_train + num_val:])
            train_IDs = np.array(train_IDs).flatten()
            val_IDs = np.array(val_IDs).flatten()
            test_IDs = np.array(test_IDs).flatten()
            self.train_dataset = NomadicLocalizationDataset(self.data_dir, train_IDs, user_positions)
            self.val_dataset = NomadicLocalizationDataset(self.data_dir, val_IDs, user_positions)
            self.test_dataset = NomadicLocalizationDataset(self.data_dir, test_IDs, user_positions)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
            #persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
            #persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
