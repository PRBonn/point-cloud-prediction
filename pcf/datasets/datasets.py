#!/usr/bin/env python3
# @brief:    Pytorch Lightning module for KITTI Odometry
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from pcf.utils.projection import projection
from pcf.utils.preprocess_data import prepare_data, compute_mean_and_std
from pcf.utils.utils import load_files


class KittiOdometryModule(LightningDataModule):
    """A Pytorch Lightning module for KITTI Odometry"""

    def __init__(self, cfg):
        """Method to initizalize the Kitti Odometry dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(KittiOdometryModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        """Call prepare_data method to generate npy range images from raw LiDAR data"""
        if self.cfg["DATA_CONFIG"]["GENERATE_FILES"]:
            prepare_data(self.cfg)

    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        ########## Point dataset splits
        train_set = KittiOdometryRaw(self.cfg, split="train")

        val_set = KittiOdometryRaw(self.cfg, split="val")

        test_set = KittiOdometryRaw(self.cfg, split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=self.cfg["DATA_CONFIG"]["DATALOADER"]["SHUFFLE"],
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=self.cfg["DATA_CONFIG"]["DATALOADER"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        # print(self.cfg["TRAIN"]["BATCH_SIZE"])
        # quit()
        self.test_iter = iter(self.test_loader)

        # Optionally compute statistics of training data
        if self.cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"]:
            compute_mean_and_std(self.cfg, self.train_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class KittiOdometryRaw(Dataset):
    """Dataset class for range image-based point cloud prediction"""

    def __init__(self, cfg, split):
        """Read parameters and scan data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("PCF_DATA_PROCESSED")
        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.n_channels = 5

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]

        # Projection class for mapping from range image to 3D point cloud
        self.projection = projection(self.cfg)

        if split == "train":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        elif split == "val":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        elif split == "test":
            self.sequences = self.cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
        else:
            raise Exception("Split must be train/val/test")

        # Create a dict filenames that maps from a sequence number to a list of files in the dataset
        self.filenames_range = {}
        self.filenames_xyz = {}
        self.filenames_intensity = {}
        self.filenames_semantic = {}

        # Create a dict idx_mapper that maps from a dataset idx to a sequence number and the index of the current scan
        self.dataset_size = 0
        self.idx_mapper = {}
        idx = 0
        for seq in self.sequences:
            seqstr = "{0:02d}".format(int(seq))
            scan_path_range = os.path.join(self.root_dir, seqstr, "processed", "range")
            self.filenames_range[seq] = load_files(scan_path_range)

            scan_path_xyz = os.path.join(self.root_dir, seqstr, "processed", "xyz")
            self.filenames_xyz[seq] = load_files(scan_path_xyz)
            assert len(self.filenames_range[seq]) == len(self.filenames_xyz[seq])

            scan_path_intensity = os.path.join(
                self.root_dir, seqstr, "processed", "intensity"
            )
            scan_path_semantic = os.path.join(
                self.root_dir, seqstr, "processed", "semantic"
            )
            self.filenames_semantic[seq] = load_files(scan_path_semantic)
            assert len(self.filenames_range[seq]) == len(self.filenames_semantic[seq])

            # Get number of sequences based on number of past and future steps
            n_samples_sequence = max(
                0,
                len(self.filenames_range[seq])
                - self.n_past_steps
                - self.n_future_steps
                + 1,
            )

            # Add to idx mapping
            for sample_idx in range(n_samples_sequence):
                scan_idx = self.n_past_steps + sample_idx - 1
                self.idx_mapper[idx] = (seq, scan_idx)
                idx += 1
            self.dataset_size += n_samples_sequence

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Load and concatenate range image channels

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        seq, scan_idx = self.idx_mapper[idx]

        # Load past data
        past_data = torch.empty(
            [self.n_channels, self.n_past_steps, self.height, self.width]
        )

        from_idx = scan_idx - self.n_past_steps + 1
        to_idx = scan_idx
        past_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        past_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        past_filenames_intensity = self.filenames_semantic[seq][from_idx : to_idx + 1]

        for t in range(self.n_past_steps):
            past_data[0, t, :, :] = self.load_range(past_filenames_range[t])
            past_data[1:4, t, :, :] = self.load_xyz(past_filenames_xyz[t])
            past_data[4, t, :, :] = self.load_intensity(past_filenames_intensity[t])

        # Load future data
        fut_data = torch.empty(
            [self.n_channels, self.n_future_steps, self.height, self.width]
        )

        from_idx = scan_idx + 1
        to_idx = scan_idx + self.n_future_steps
        fut_filenames_range = self.filenames_range[seq][from_idx : to_idx + 1]
        fut_filenames_xyz = self.filenames_xyz[seq][from_idx : to_idx + 1]
        fut_filenames_intensity = self.filenames_semantic[seq][from_idx : to_idx + 1]

        for t in range(self.n_future_steps):
            fut_data[0, t, :, :] = self.load_range(fut_filenames_range[t])
            fut_data[1:4, t, :, :] = self.load_xyz(fut_filenames_xyz[t])
            fut_data[4, t, :, :] = self.load_intensity(fut_filenames_intensity[t])

        item = {"past_data": past_data, "fut_data": fut_data, "meta": (seq, scan_idx)}
        return item

    def load_range(self, filename):
        """Load .npy range image as (1,height,width) tensor"""
        rv = torch.Tensor(np.load(filename)).float()
        return rv

    def load_xyz(self, filename):
        """Load .npy xyz values as (3,height,width) tensor"""
        xyz = torch.Tensor(np.load(filename)).float()[:, :, :3]
        xyz = xyz.permute(2, 0, 1)
        return xyz

    def load_intensity(self, filename):
        """Load .npy intensity values as (1,height,width) tensor"""
        intensity = torch.Tensor(np.load(filename)).float()
        return intensity


if __name__ == "__main__":
    config_filename = "./config/parameters.yml"
    cfg = yaml.safe_load(open(config_filename))
    data = KittiOdometryModule(cfg)
    data.prepare_data()
    data.setup()

    item = data.valid_loader.dataset.__getitem__(0)

    def normalize(image):
        min = np.min(image)
        max = np.max(image)
        normalized_image = (image - min) / (max - min)
        return normalized_image

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(30, 30 * 5 * 64 / 2048))

    axs[0].imshow(normalize(item["fut_data"][0, 0, :, :].numpy()))
    axs[0].set_title("Range")
    axs[1].imshow(normalize(item["fut_data"][1, 0, :, :].numpy()))
    axs[1].set_title("X")
    axs[2].imshow(normalize(item["fut_data"][2, 0, :, :].numpy()))
    axs[2].set_title("Y")
    axs[3].imshow(normalize(item["fut_data"][3, 0, :, :].numpy()))
    axs[3].set_title("Z")
    axs[4].imshow(normalize(item["fut_data"][4, 0, :, :].numpy()))
    axs[4].set_title("Intensity")

    plt.show()
