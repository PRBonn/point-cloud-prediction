#!/usr/bin/env python3
from functools import partial
import os
import glob
import time
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 1000
HEIGHT = 1000
POINTSIZE = 1.5
SLEEPTIME = 0.3


def get_car_model(filename):
    """Car model for visualization

    Args:
        filename (str): filename of mesh

    Returns:
        mesh: open3D mesh
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def get_filename(path, idx):
    filenames = sorted(glob.glob(path + "*.ply"))
    return int(filenames[idx].split(".")[0].split("/")[-1])


def last_file(path):
    return get_filename(path, -1)


def first_file(path):
    return get_filename(path, 0)


class Visualization:
    """Visualization of point cloud predictions with open3D"""

    def __init__(
        self,
        path,
        sequence,
        start,
        end,
        capture=False,
        path_to_car_model=None,
        sleep_time=5e-3,
    ):
        """Init

        Args:
            path (str): path to data should be
              .
              ├── sequence
              │   ├── gt
              |   |   ├──frame.ply
              │   ├─── pred
              |   |   ├── frame
              |   │   |   ├─── (frame+1).ply

            sequence (int): Sequence to visualize
            start (int): Start at specific frame
            end (itn): End at specific frame
            capture (bool, optional): Save to file at each frame. Defaults to False.
            path_to_car_model (str, optional): Path to car model. Defaults to None.
            sleep_time (float, optional): Sleep time between frames. Defaults to 5e-3.
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=WIDTH, height=HEIGHT)
        self.render_options = self.vis.get_render_option()
        self.render_options.point_size = POINTSIZE
        self.capture = capture

        # Load car model
        if path_to_car_model:
            self.car_mesh = get_car_model(path_to_car_model)
        else:
            self.car_mesh = None

        # Path and sequence to visualize
        self.path = path
        self.sequence = sequence

        # Frames to visualize
        self.start = start
        self.end = end

        # Init
        self.current_frame = self.start
        self.current_step = 1
        self.n_pred_steps = 5

        # Save last view
        self.ctr = self.vis.get_view_control()
        self.camera = self.ctr.convert_to_pinhole_camera_parameters()
        self.viewpoint_path = os.path.join(self.path, "viewpoint.json")

        self.print_help()
        self.update(self.vis)

        # Continuous time plot
        self.stop = False
        self.sleep_time = sleep_time

        # Initialize the default callbacks
        self._register_key_callbacks()

        self.last_time_key_pressed = time.time()

    def prev_frame(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_frame = max(self.start, self.current_frame - 1)
            self.update(vis)
        return False

    def next_frame(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_frame = min(self.end, self.current_frame + 1)
            self.update(vis)
        return False

    def prev_prediction_step(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_step = max(1, self.current_step - 1)
            self.update(vis)
        return False

    def next_prediction_step(self, vis):
        if time.time() - self.last_time_key_pressed > SLEEPTIME:
            self.last_time_key_pressed = time.time()
            self.current_step = min(self.n_pred_steps, self.current_step + 1)
            self.update(vis)
        return False

    def play_sequence(self, vis):
        self.stop = False
        while not self.stop:
            self.next_frame(vis)
            time.sleep(self.sleep_time)

    def stop_sequence(self, vis):
        self.stop = True

    def toggle_capture_mode(self, vis):
        if self.capture:
            self.capture = False
        else:
            self.capture = True

    def update(self, vis):
        """Get point clouds and visualize"""
        print(
            "Current frame: {:d}/{:d}, Prediction step {:d}/{:d}, capture_mode: {:b}".format(
                self.current_frame,
                self.end,
                self.current_step,
                self.n_pred_steps,
                self.capture,
            ),
            end="\r",
        )
        gt_pcd, pred_pcd = self.get_gt_and_predictions(
            self.path, self.sequence, self.current_frame, self.current_step
        )

        geometry_list = [gt_pcd, pred_pcd]
        if self.car_mesh:
            geometry_list.append(self.car_mesh)
        self.vis_update_geometries(vis, geometry_list)

        if self.capture:
            self.capture_frame()

    def get_gt_and_predictions(self, path, sequence, current_frame, step):
        """Load GT and predictions from path

        Args:
            path (str): Path to files
            sequence (int): Sequence to visualize
            current_frame (int): Last received frame for prediction
            step (int): Prediction step to visualize

        Returns:
            o3d.point_cloud: GT and predicted point clouds
        """
        pred_path = os.path.join(
            path,
            sequence,
            "pred",
            str(current_frame).zfill(6),
            str(current_frame + step).zfill(6) + ".ply",
        )
        pred_pcd = o3d.io.read_point_cloud(pred_path)
        pred_pcd.paint_uniform_color([0, 0, 1])

        gt_path = os.path.join(
            path, sequence, "gt", str(current_frame + step).zfill(6) + ".ply"
        )
        gt_pcd = o3d.io.read_point_cloud(gt_path)
        gt_pcd.paint_uniform_color([1, 0, 0])
        return gt_pcd, pred_pcd

    def vis_update_geometries(self, vis, geometries):
        """Save camera pose and update point clouds"""
        # Save current camera pose
        self.camera = self.ctr.convert_to_pinhole_camera_parameters()

        vis.clear_geometries()
        for geometry in geometries:
            vis.add_geometry(geometry)

        # Set to last view
        self.ctr.convert_from_pinhole_camera_parameters(self.camera)

        self.vis.poll_events()
        self.vis.update_renderer()

    def set_render_options(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.render_options, key, value)

    def register_key_callback(self, key, callback):
        self.vis.register_key_callback(ord(str(key)), partial(callback))

    def set_white_background(self, vis):
        """Change backround between white and white"""
        self.render_options.background_color = [1.0, 1.0, 1.0]

    def set_black_background(self, vis):
        """Change backround between white and black"""
        self.render_options.background_color = [0.0, 0.0, 0.0]

    def save_viewpoint(self, vis):
        """Saves viewpoint"""
        self.camera = self.ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(self.viewpoint_path, self.camera)

    def load_viewpoint(self, vis):
        """Loads viewpoint"""
        self.camera = o3d.io.read_pinhole_camera_parameters(self.viewpoint_path)
        self.ctr.convert_from_pinhole_camera_parameters(self.camera)

    def _register_key_callbacks(self):
        self.register_key_callback("L", self.next_frame)
        self.register_key_callback("H", self.prev_frame)
        self.register_key_callback("K", self.next_prediction_step)
        self.register_key_callback("J", self.prev_prediction_step)
        self.register_key_callback("S", self.play_sequence)
        self.register_key_callback("X", self.stop_sequence)
        self.register_key_callback("C", self.toggle_capture_mode)
        self.register_key_callback("W", self.set_white_background)
        self.register_key_callback("B", self.set_black_background)
        self.register_key_callback("V", self.save_viewpoint)
        self.register_key_callback("Q", self.load_viewpoint)

    def print_help(self):
        print("L: next frame")
        print("H: previous frame")
        print("K: next prediction step")
        print("J: previous prdeiction step")
        print("S: start")
        print("X: stop")
        print("C: Toggle capture mode")
        print("W: white background")
        print("B: black  background")
        print("V: save viewpoint")
        print("Q: set to saved viewpoint")
        print("ESC: quit")

    def capture_frame(self):
        """Save view from current viewpoint"""
        image = self.vis.capture_screen_float_buffer(False)
        path = os.path.join(self.path, self.sequence, "images")
        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(
            path,
            "step_{:1d}_prediction_from_frame_{:05d}.png".format(
                self.current_step, self.current_frame
            ),
        )
        print("Capture image ", filename)
        plt.imsave(filename, np.asarray(image), dpi=1)

    def run(self):
        self.vis.run()
        self.vis.destroy_window()
