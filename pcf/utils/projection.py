#!/usr/bin/env python3
# @brief:    Get a 3D point cloud from a given range image projection
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import torch
import torch.nn as nn
import numpy as np


class projection:
    """Projection class for getting a 3D point cloud from range images"""

    def __init__(self, cfg):
        """Init

        Args:
            cfg (dict): Parameters
        """
        self.cfg = cfg

        fov_up = (
            self.cfg["DATA_CONFIG"]["FOV_UP"] / 180.0 * np.pi
        )  # field of view up in radians
        fov_down = (
            self.cfg["DATA_CONFIG"]["FOV_DOWN"] / 180.0 * np.pi
        )  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radian
        W = self.cfg["DATA_CONFIG"]["WIDTH"]
        H = self.cfg["DATA_CONFIG"]["HEIGHT"]

        h = torch.arange(0, H).view(H, 1).repeat(1, W)
        w = torch.arange(0, W).view(1, W).repeat(H, 1)
        yaw = np.pi * (1.0 - 2 * torch.true_divide(w, W))
        pitch = (1.0 - torch.true_divide(h, H)) * fov - abs(fov_down)
        self.x_fac = torch.cos(pitch) * torch.cos(yaw)
        self.y_fac = torch.cos(pitch) * torch.sin(yaw)
        self.z_fac = torch.sin(pitch)

    def get_valid_points_from_range_view(self, range_view):
        """Reproject from range image to valid 3D points

        Args:
            range_view (torch.tensor): Range image with size (H,W)

        Returns:
            torch.tensor: Valid 3D points with size (N,3)
        """
        H, W = range_view.shape
        points = torch.zeros(H, W, 3).type_as(range_view)
        points[:, :, 0] = range_view * self.x_fac.type_as(range_view)
        points[:, :, 1] = range_view * self.y_fac.type_as(range_view)
        points[:, :, 2] = range_view * self.z_fac.type_as(range_view)
        return points[range_view > 0.0]

    def get_mask_from_output(self, output):
        """Get mask from logits

        Args:
            output (dict): Output dict with mask_logits as key

        Returns:
            mask: Predicted mask containing per-point probabilities
        """
        mask = nn.Sigmoid()(output["mask_logits"])
        return mask

    def get_target_mask_from_range_view(self, range_view):
        """Ground truth mask

        Args:
            range_view (torch.tensor): Range image of size (H,W)

        Returns:
            torch.tensor: Target mask of valid points
        """
        target_mask = torch.zeros(range_view.shape).type_as(range_view)
        target_mask[range_view != -1.0] = 1.0
        return target_mask

    def get_masked_range_view(self, output):
        """Get predicted masked range image

        Args:
            output (dict): Dictionary containing predicted mask logits and ranges

        Returns:
            torch.tensor: Maskes range image in which invalid points are mapped to -1.0
        """
        mask = self.get_mask_from_output(output)
        masked_range_view = output["rv"].clone()

        # Set invalid points to -1.0 according to mask
        masked_range_view[mask < self.cfg["MODEL"]["MASK_THRESHOLD"]] = -1.0
        return masked_range_view
