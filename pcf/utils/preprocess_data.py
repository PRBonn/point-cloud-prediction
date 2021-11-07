#!/usr/bin/env python3
# @brief:    Preprocessing point cloud to range images
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import os
import numpy as np
import torch

from pcf.utils.utils import load_files, range_projection


def prepare_data(cfg):
    """Loads point clouds and pre-processes them into range images

    Args:
        cfg (dict): Config
    """
    sequences = (
        cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        + cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        + cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
    )

    for seq in sequences:
        seqstr = "{0:02d}".format(int(seq))
        scan_folder = os.path.join(os.environ.get("PCF_DATA_RAW"), seqstr, "velodyne")
        dst_folder = os.path.join(
            os.environ.get("PCF_DATA_PROCESSED"), seqstr, "processed"
        )
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Load LiDAR scan files
        scan_paths = load_files(scan_folder)

        # Iterate over all scan files
        for idx in range(len(scan_paths)):
            print(
                "Processing file {:d}/{:d} of sequence {:d}".format(
                    idx, len(scan_paths), seq
                )
            )

            # Load and project a point cloud
            current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
            current_vertex = current_vertex.reshape((-1, 4))
            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                current_vertex,
                fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )

            # Save range
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(6))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(6))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(6))
            np.save(file_path, proj_intensity)


def compute_mean_and_std(cfg, train_loader):
    """Compute training data statistics

    Args:
        cfg (dict): Config
        train_loader (DataLoader): Pytorch DataLoader to access training data
    """
    n_channels = train_loader.dataset.n_channels
    mean = [0] * n_channels
    std = [0] * n_channels
    max = [0] * n_channels
    min = [0] * n_channels
    for i, data in enumerate(train_loader):
        past = data["past_data"]
        batch_size, n_channels, frames, H, W = past.shape

        for j in range(n_channels):
            channel = past[:, j, :, :, :].view(batch_size, 1, frames, H, W)
            mean[j] += torch.mean(channel[channel != -1.0]) / len(train_loader)
            std[j] += torch.std(channel[channel != -1.0]) / len(train_loader)
            max[j] += torch.max(channel[channel != -1.0]) / len(train_loader)
            min[j] += torch.min(channel[channel != -1.0]) / len(train_loader)

    print("Mean and standard deviation of training data:")
    for j in range(n_channels):
        print(
            "Input {:d}: Mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
                j, mean[j], std[j], min[j], max[j]
            )
        )
