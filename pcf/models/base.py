#!/usr/bin/env python3
# @brief:    Point cloud prediction architecture with 3D Spatio-Temporal Convolutional Networks
# @author    Benedikt Mersch    [mersch@igg.uni-bonn.de]
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning.core.module import LightningModule
from pcf.models.loss import Loss
from pcf.utils.projection import projection
from pcf.utils.logger import log_point_clouds, save_range_and_mask, save_point_clouds
from lion_pytorch import Lion

class BasePredictionModel(LightningModule):
    """Pytorch Lightning base model for point cloud prediction"""

    def __init__(self, cfg):
        """Init base model

        Args:
            cfg (dict): Config parameters
        """
        super(BasePredictionModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        self.height = self.cfg["DATA_CONFIG"]["HEIGHT"]
        self.width = self.cfg["DATA_CONFIG"]["WIDTH"]
        self.min_range = self.cfg["DATA_CONFIG"]["MIN_RANGE"]
        self.max_range = self.cfg["DATA_CONFIG"]["MAX_RANGE"]
        self.register_buffer("mean", torch.Tensor(self.cfg["DATA_CONFIG"]["MEAN"]))
        self.register_buffer("std", torch.Tensor(self.cfg["DATA_CONFIG"]["STD"]))

        self.n_past_steps = self.cfg["MODEL"]["N_PAST_STEPS"]
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.use_xyz = self.cfg["MODEL"]["USE"]["XYZ"]
        self.use_intensity = self.cfg["MODEL"]["USE"]["INTENSITY"]

        # Create list of index used in input
        self.inputs = [0]
        if self.use_xyz:
            self.inputs.append(1)
            self.inputs.append(2)
            self.inputs.append(3)
        if self.use_intensity:
            self.inputs.append(4)
        self.n_inputs = len(self.inputs)

        # Init loss
        self.loss = Loss(self.cfg)

        # Init projection class for re-projcecting from range images to 3D point clouds
        self.projection = projection(self.cfg)

        self.chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        """Optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        #optimizer = Lion(self.parameters(), lr=self.cfg["TRAIN"]["LR"])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg["TRAIN"]["LR_EPOCH"],
            gamma=self.cfg["TRAIN"]["LR_DECAY"],
        )
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        #                        T_max = 400, # Maximum number of iterations.
        #                        eta_min = 1e-5, verbose= True)
        return [optimizer], [scheduler]
        #return {
        #        'optimizer': optimizer,
        #        'lr_scheduler': scheduler,
        #        'monitor': 'val/loss'
        #    }

    def training_step(self, batch, batch_idx):
        """Pytorch Lightning training step including logging

        Args:
            batch (dict): A dict with a batch of training samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        past = batch["past_data"]
        future = batch["fut_data"]
        output = self.forward(past)
        #print(self.current_epoch)
        loss = self.loss(output, future, "train")
        #print(loss["loss"].item())
        self.log("train/loss", loss["loss"])
        self.log("train/mean_chamfer_distance", loss["mean_chamfer_distance"])
        self.log("train/final_chamfer_distance", loss["final_chamfer_distance"])
        self.log("train/loss_range_view", loss["loss_range_view"])
        self.log("train/loss_mask", loss["loss_mask"])

        return loss

    def validation_step(self, batch, batch_idx):
        """Pytorch Lightning validation step including logging

        Args:
            batch (dict): A dict with a batch of validation samples
            batch_idx (int): Index of batch in dataset

        Returns:
            None
        """
        past = batch["past_data"]
        future = batch["fut_data"]
        output = self.forward(past)

        loss = self.loss(output, future, "val")

        self.log("val/loss", loss["loss"], on_epoch=True)
        self.log(
            "val/mean_chamfer_distance", loss["mean_chamfer_distance"], on_epoch=True
        )
        self.log(
            "val/final_chamfer_distance", loss["final_chamfer_distance"], on_epoch=True
        )
        self.log("val/loss_range_view", loss["loss_range_view"], on_epoch=True)
        self.log("val/loss_mask", loss["loss_mask"], on_epoch=True)

        #selected_sequence_and_frame = self.cfg["VALIDATION"][
        #    "SELECTED_SEQUENCE_AND_FRAME"
        #]
        #sequence_batch, frame_batch = batch["meta"]
        #for sample_idx in range(frame_batch.shape[0]):
        #    sequence = sequence_batch[sample_idx].item()
        #    frame = frame_batch[sample_idx].item()
        #    if sequence in selected_sequence_and_frame.keys():
        #        if frame in selected_sequence_and_frame[sequence]:
        #            t1 = time.time()
        #            log_point_clouds(
        #                self.logger.experiment,
        #                self.projection,
        #                self.current_epoch,
        #                batch,
        #                output,
        #                sample_idx,
        #                sequence,
        #                frame,
        #            )
        #            print("log_point_clouds: ",time.time()-t1)
        #            t1 = time.time()
        #            save_range_and_mask(
        #                self.cfg,
        #                self.projection,
        #                batch,
        #                output,
        #                sample_idx,
        #                sequence,
        #                frame,
        #            )
        #            print("save_range_and_mask: ",time.time()-t1)

    def test_step(self, batch, batch_idx):
        """Pytorch Lightning test step including logging

        Args:
            batch (dict): A dict with a batch of test samples
            batch_idx (int): Index of batch in dataset

        Returns:
            loss (dict): Multiple loss components
        """
        print("Test")
        past = batch["past_data"]
        future = batch["fut_data"]

        batch_size, n_inputs, n_future_steps, H, W = past.shape

        start = time.time()
        output = self.forward(past)
        inference_time = (time.time() - start) / batch_size
        self.log("test/inference_time", inference_time, on_epoch=True)

        loss = self.loss(output, future, "test")

        self.log("test/loss_range_view", loss["loss_range_view"], on_epoch=True)
        self.log("test/loss_mask", loss["loss_mask"], on_epoch=True)

        for step, value in loss["chamfer_distance"].items():
            self.log("test/chamfer_distance_{:d}".format(step), value, on_epoch=True)

        self.log(
            "test/mean_chamfer_distance", loss["mean_chamfer_distance"], on_epoch=True
        )
        self.log(
            "test/final_chamfer_distance", loss["final_chamfer_distance"], on_epoch=True
        )

        self.chamfer_distances_tensor = torch.cat(
            (self.chamfer_distances_tensor, loss["chamfer_distances_tensor"]), dim=1
        )

        #print(self.cfg["TEST"]["SAVE_POINT_CLOUDS"])
        if self.cfg["TEST"]["SAVE_POINT_CLOUDS"]:
            #save_point_clouds(self.cfg, self.projection, batch, output)

            sequence_batch, frame_batch = batch["meta"]
            for sample_idx in range(frame_batch.shape[0]):
                sequence = sequence_batch[sample_idx].item()
                frame = frame_batch[sample_idx].item()
                save_range_and_mask(
                    self.cfg,
                    self.projection,
                    batch,
                    output,
                    sample_idx,
                    sequence,
                    frame,
                )

        return loss

    def test_epoch_end(self, outputs):
        # Remove first row since it was initialized with zero
        self.chamfer_distances_tensor = self.chamfer_distances_tensor[:, 1:]
        n_steps, _ = self.chamfer_distances_tensor.shape
        mean = torch.mean(self.chamfer_distances_tensor, dim=1)
        std = torch.std(self.chamfer_distances_tensor, dim=1)
        q = torch.tensor([0.25, 0.5, 0.75])
        quantile = torch.quantile(self.chamfer_distances_tensor, q, dim=1)

        chamfer_distances = []
        for s in range(n_steps):
            chamfer_distances.append(self.chamfer_distances_tensor[s, :].tolist())
        print("Final size of CD: ", self.chamfer_distances_tensor.shape)
        print("Mean :", mean)
        print("Std :", std)
        print("Quantile :", quantile)

        testdir = os.path.join(self.cfg["LOG_DIR"], "test")
        if not os.path.exists(testdir):
            os.makedirs(testdir)

        filename = os.path.join(
            testdir, "stats_" + time.strftime("%Y%m%d_%H%M%S") + ".yml"
        )

        log_to_save = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "quantile": quantile.tolist(),
            "chamfer_distances": chamfer_distances,
        }
        with open(filename, "w") as yaml_file:
            yaml.dump(log_to_save, yaml_file, default_flow_style=False)
