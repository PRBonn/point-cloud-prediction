import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random

from pyTorchChamferDistance.chamfer_distance import ChamferDistance
from pcf.utils.projection import projection


class Loss(nn.Module):
    """Combined loss for point cloud prediction"""

    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.loss_weight_cd = self.cfg["TRAIN"]["LOSS_WEIGHT_CHAMFER_DISTANCE"]
        self.loss_weight_rv = self.cfg["TRAIN"]["LOSS_WEIGHT_RANGE_VIEW"]
        self.loss_weight_mask = self.cfg["TRAIN"]["LOSS_WEIGHT_MASK"]

        self.loss_range = loss_range(self.cfg)
        self.chamfer_distance = chamfer_distance(self.cfg)
        self.loss_mask = loss_mask(self.cfg)

    def forward(self, output, target, mode):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """

        target_range_image = target[:, 0, :, :, :]
        target_semantic_label = target[:, 4, :, :, :]
        
        ground_mask = (
                (target_semantic_label==70) | (target_semantic_label==40)\
                        | (target_semantic_label==44) | (target_semantic_label==48)\
                        | (target_semantic_label==49) | (target_semantic_label==50)\
                ).type(torch.uint8)
        object_mask = torch.logical_not(ground_mask)

        # Range view
        loss_range_view = self.loss_range(output, target_range_image, ground_mask, object_mask)

        # Mask
        loss_mask = self.loss_mask(output, target_range_image)

        # Chamfer Distance
        if self.loss_weight_cd > 0.0 or mode == "val" or mode == "test":
            chamfer_distance, chamfer_distances_tensor = self.chamfer_distance(
                output, target, self.cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"]
            )
            loss_chamfer_distance = sum([cd for cd in chamfer_distance.values()]) / len(
                chamfer_distance
            )
            detached_chamfer_distance = {
                step: cd.detach() for step, cd in chamfer_distance.items()
            }
        else:
            chamfer_distance = dict(
                (step, torch.zeros(1).type_as(target_range_image))
                for step in range(self.n_future_steps)
            )
            chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
            loss_chamfer_distance = torch.zeros_like(loss_range_view)
            detached_chamfer_distance = chamfer_distance

        loss = (
            self.loss_weight_cd * loss_chamfer_distance
            + self.loss_weight_rv * loss_range_view
            + self.loss_weight_mask * loss_mask
        )

        loss_dict = {
            "loss": loss,
            "chamfer_distance": detached_chamfer_distance,
            "chamfer_distances_tensor": chamfer_distances_tensor.detach(),
            "mean_chamfer_distance": loss_chamfer_distance.detach(),
            "final_chamfer_distance": chamfer_distance[
                self.n_future_steps - 1
            ].detach(),
            "loss_range_view": loss_range_view.detach(),
            "loss_mask": loss_mask.detach(),
        }
        return loss_dict


class loss_mask(nn.Module):
    """Binary cross entropy loss for prediction of valid mask"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.projection = projection(self.cfg)

    def forward(self, output, target_range_view):
        target_mask = self.projection.get_target_mask_from_range_view(target_range_view)
        loss = self.loss(output["mask_logits"], target_mask)
        return loss


class loss_range(nn.Module):
    """L1 loss for range image prediction"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, output, target_range_image, ground_mask, object_mask):
        # Do not count L1 loss for invalid GT points
        gt_masked_output = output["rv"].clone()
        gt_masked_output[target_range_image == -1.0] = -1.0
        #print((non_veg_ranges*gt_masked_output).shape)
        #quit()

        loss = self.loss(gt_masked_output, target_range_image)
        #loss = self.loss(ground_mask*gt_masked_output, ground_mask*target_range_image)\
        #        + 2*self.loss(object_mask*gt_masked_output, object_mask*target_range_image)
        return loss


class chamfer_distance(nn.Module):
    """Chamfer distance loss. Additionally, the implementation allows the evaluation
    on downsampled point cloud (this is only for comparison to other methods but not recommended,
    because it is a bad approximation of the real Chamfer distance.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target, n_samples):
        batch_size, n_future_steps, H, W = output["rv"].shape
        masked_output = self.projection.get_masked_range_view(output)
        chamfer_distances = {}
        chamfer_distances_tensor = torch.zeros(n_future_steps, batch_size)
        for s in range(n_future_steps):
            chamfer_distances[s] = 0
            for b in range(batch_size):
                output_points = self.projection.get_valid_points_from_range_view(
                    masked_output[b, s, :, :]
                ).view(1, -1, 3)
                target_points = target[b, 1:4, s, :, :].permute(1, 2, 0)
                target_points = target_points[target[b, 0, s, :, :] > 0.0].view(
                    1, -1, 3
                )

                if n_samples != -1:
                    n_output_points = output_points.shape[1]
                    n_target_points = target_points.shape[1]

                    sampled_output_indices = random.sample(
                        range(n_output_points), n_samples
                    )
                    sampled_target_indices = random.sample(
                        range(n_target_points), n_samples
                    )

                    output_points = output_points[:, sampled_output_indices, :]
                    target_points = target_points[:, sampled_target_indices, :]

                dist1, dist2 = self.loss(output_points, target_points)
                dist_combined = torch.mean(dist1) + torch.mean(dist2)
                chamfer_distances[s] += dist_combined
                chamfer_distances_tensor[s, b] = dist_combined
            chamfer_distances[s] = chamfer_distances[s] / batch_size
        return chamfer_distances, chamfer_distances_tensor
