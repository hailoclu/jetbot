#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Helper class for obtaining groundtruth data from OmniKit.

Support provided for RGB, Depth, Bounding Box (2D Tight, 2D Loose, 3D),
segmentation (instance and semantic), and camera parameters.

    Typical usage example:

    kit = OmniKitHelper()   # Start omniverse kit
    sd_helper = SyntheticDataHelper()
    gt = sd_helper.get_groundtruth(('rgb', 'depth', 'boundingBox2DTight'))

"""

import math
import carb
import omni.kit.editor
from pxr import UsdGeom, Semantics, Gf

import numpy as np


try:
    import torch
    import torch_wrap
    from . import utils

    use_torch = True
except ImportError as err:
    carb.log_info(f"Torch could not be imported: {err}")
    use_torch = False

from .camera import get_view_proj_mat


# List of sensors requiring RayTracedLighting mode
RT_ONLY_SENSORS = []


class SyntheticDataHelper:
    def __init__(self):
        self.app = omni.kit.app.get_app_interface()
        ext_manager = self.app.get_extension_manager()
        ext_manager.set_extension_enabled("omni.syntheticdata", True)

        import omni.syntheticdata._syntheticdata as sd  # Must be imported after getting app interface

        self.sd = sd

        self.sd_interface = self.sd.acquire_syntheticdata_interface()
        self.editor = omni.kit.editor.get_editor_interface()
        self.carb_settings = carb.settings.acquire_settings_interface()

        # mode = 'cuda' if use_torch else 'numpy'

        # Currently only numpy output is supported
        mode = "numpy"

        self.sensor_helpers = {
            "rgb": getattr(self, f"get_rgb_{mode}"),
            "depth": getattr(self, f"get_depth_{mode}"),
            "depthLinear": getattr(self, f"get_depth_linear_{mode}"),
            "instanceSegmentation": getattr(self, f"get_instance_segmentation_{mode}"),
            "semanticSegmentation": getattr(self, f"get_semantic_segmentation_{mode}"),
            "boundingBox2DTight": self.get_bbox2d_tight,
            "boundingBox2DLoose": self.get_bbox2d_loose,
            "boundingBox3D": self.compute_3d_bounding_box_oobb,
            "camera": self.get_camera_params,
            "pose": self.get_pose,
        }

        self.sensor_state = {s: False for s in list(self.sensor_helpers.keys())}

    def enable_sensors(self, sensors):
        """Enable syntheticdata sensors.

        Args:
            sensors: List of sensor names. Valid sensors names: rgb, depth,
                     instanceSegmentation, semanticSegmentation, boundingBox2DTight,
                     boundingBox2DLoose, boundingBox3D, camera
        """
        if isinstance(sensors, str):
            sensors = (sensors,)
        for sensor in sensors:
            if sensor not in list(self.sensor_helpers.keys()):
                raise ValueError
            if not self.sensor_state[sensor]:
                self.carb_settings.set_bool(f"/syntheticdata/sensors/{sensor}Sensor", True)
                self.sensor_state[sensor] = True
                for _ in range(2):
                    self.app.update(0.0)  # sensor buffers need two frames to be correctly populated.

    def get_rgb_cuda(self):
        """Get RGB groundtruth data

        Returns:
            Tensor(dtype=uint8, shape=(H, W, C))
        """
        return self._get_sensor_cuda_tensor(self.sd.SensorType.Rgb, "uint8")

    def get_rgb_numpy(self):
        """Get RGB groundtruth data

        Returns:
            numpy.ndarray(dtype=uint8, shape=(H, W, C))
        """
        data = self._get_sensor_data(self.sd.SensorType.Rgb, "uint32")
        image_data = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
        return image_data

    def get_depth_cuda(self):
        """Get Depth groundtruth data

        Returns:
            Tensor(dtype=float32, shape=(H, W))
        """
        depth = self._get_sensor_cuda_tensor(self.sd.SensorType.Depth, "float")
        return depth

    def get_depth_numpy(self):
        """Get Depth groundtruth data

        Returns:
            numpy.ndarray(dtype=float32, shape=(H, W))
        """
        depth = self._get_sensor_data(self.sd.SensorType.Depth, "float")
        return depth

    def get_depth_linear_cuda(self):
        """Get Linear Depth groundtruth data

        Returns:
            Tensor(dtype=float32, shape=(H, W))
        """
        depthLinear = self._get_sensor_cuda_tensor(self.sd.SensorType.DepthLinear, "float")
        return depthLinear

    def get_depth_linear_numpy(self):
        """Get Linear Depth groundtruth data

        Returns:
            numpy.ndarray(dtype=float32, shape=(H, W))
        """
        depthLinear = self._get_sensor_data(self.sd.SensorType.DepthLinear, "float")
        return depthLinear

    def _get_sensor_data(self, sensor, dtype):
        width = self.sd_interface.get_sensor_width(sensor)
        height = self.sd_interface.get_sensor_height(sensor)
        row_size = self.sd_interface.get_sensor_row_size(sensor)

        get_sensor = {
            "uint32": self.sd_interface.get_sensor_host_uint32_texture_array,
            "float": self.sd_interface.get_sensor_host_float_texture_array,
        }
        return get_sensor[dtype](sensor, width, height, row_size)

    def _get_sensor_cuda_tensor(self, sensor, dtype):
        width = self.sd_interface.get_sensor_width(sensor)
        height = self.sd_interface.get_sensor_height(sensor)
        row_size = self.sd_interface.get_sensor_row_size(sensor)

        get_sensor = {
            "int32": self.sd_interface.get_sensor_device_int32_2d_tensor,
            "float": self.sd_interface.get_sensor_device_float_2d_tensor,
            "uint8": self.sd_interface.get_sensor_device_uint8_3d_tensor,
        }
        tensor_data = get_sensor[dtype](sensor, height, width, row_size)
        return torch_wrap.wrap_tensor(tensor_data)

    def get_instance_segmentation_cuda(self):
        """Get instance segmentation data.
        Generate a list of N instance names and corresponding array of N
        binary instance masks.

        Returns:
            A tuple of a list of instance names, and a bool CUDA tensor with shape (N, H, W).
        """
        sensor = self.sd.SensorType.InstanceSegmentation
        instance_tex = self._get_sensor_cuda_tensor(sensor, "int32")
        instance_mappings = self.get_instance_mappings()
        instances_list = [im[4] for im in instance_mappings]
        instance_names = [im[0] for im in instance_mappings]

        instance_masks = torch.zeros((len(instance_mappings), *instance_tex.shape), dtype=np.bool, device="cuda")
        for i, instances in enumerate(instances_list):
            instance_masks[i] = utils.torch_isin(instance_tex, torch.tensor(instances).to(instance_tex))
        return instance_names, instance_masks

    def get_instance_segmentation_numpy(self):
        """Get instance segmentation data.
        Generate a list of N instance names and corresponding array of N
        binary instance masks.

        Returns:
            A tuple of a list of instance names, and a bool array with shape (N, H, W).
        """
        sensor = self.sd.SensorType.InstanceSegmentation
        instance_tex = self._get_sensor_data(sensor, "uint32")
        instance_mappings = self.get_instance_mappings()
        instances_list = [im[3] for im in instance_mappings]
        instance_names = [im[0] for im in instance_mappings]

        instance_masks = np.zeros((len(instance_mappings), *instance_tex.shape), dtype=np.bool)
        for i, instances in enumerate(instances_list):
            instance_masks[i] = np.isin(instance_tex, instances)
        return instance_names, instance_masks

    def get_semantic_segmentation_cuda(self):
        """ Get semantic segmentation data.
        Generate a list of N semantic labels and corresponding array of N
        binary semantic masks.

        Returns:
            A tuple of a list of semantic labels, and a bool CUDA tensor with shape (N, H, W).
        """
        sensor = self.sd.SensorType.InstanceSegmentation
        instance_seg_texture = self._get_sensor_cuda_tensor(sensor, "int32")
        instance_mappings = self.get_instance_mappings()

        semantic_mappings = {}
        for im in instance_mappings:
            semantic_mappings.setdefault(im[2], []).extend(im[3])

        semantic_labels = list(semantic_mappings.keys())
        semantic_masks = torch.zeros((len(semantic_labels), *instance_seg_texture.shape), dtype=np.bool)
        for i, (semantic_label, instances) in enumerate(semantic_mappings.items()):
            semantic_masks[i] = utils.torch_isin(instance_seg_texture, torch.tensor(instances).to(instance_seg_texture))
        return semantic_labels, semantic_masks

    def get_semantic_segmentation_numpy(self):
        """ Get semantic segmentation data.
        Generate a list of N semantic labels and corresponding array of N
        binary semantic masks.

        Returns:
            A tuple of a list of semantic labels, and a bool array with shape (N, H, W).
        """
        sensor = self.sd.SensorType.InstanceSegmentation
        instance_seg_texture = self._get_sensor_data(sensor, "uint32")
        instance_mappings = self.get_instance_mappings()

        semantic_mappings = {}
        for im in instance_mappings:
            semantic_mappings.setdefault(im[2], []).extend(im[3])

        semantic_labels = list(semantic_mappings.keys())
        semantic_masks = np.zeros((len(semantic_labels), *instance_seg_texture.shape), dtype=np.bool)
        for i, (semantic_label, instances) in enumerate(semantic_mappings.items()):
            semantic_masks[i] = np.isin(instance_seg_texture, instances)
        return semantic_labels, semantic_masks

    def get_bbox2d_tight(self):
        """Get Bounding Box 2D with tight bounds.
        Bounding boxes will only bound visible portions of an object.

        Returns:
            numpy.ndarray(dtype=[('semanticLabel', 'O'), ('instanceId', '<u4'),
                                 ('semanticId', '<u4'), ('x_min', '<i4'), ('y_min', '<i4'),
                                 ('x_max', '<i4'), ('y_max', '<i4')]),
                          shape=N)
        """
        sensor = self.sd.SensorType.BoundingBox2DTight
        return self._get_bbox2d(sensor)

    def get_bbox2d_loose(self):
        """Get Bounding Box 2D with tight bounds.
        Bounding boxes will bound entire object, including occluded portions.

        Returns:
            numpy.ndarray(dtype=[('semanticLabel', 'O'), ('instanceId', '<u4'),
                                 ('semanticId', '<u4'), ('x_min', '<i4'), ('y_min', '<i4'),
                                 ('x_max', '<i4'), ('y_max', '<i4')]),
                          shape=N)
        """
        sensor = self.sd.SensorType.BoundingBox2DLoose
        return self._get_bbox2d(sensor)

    def get_camera_params(self):
        """Get active camera intrinsic and extrinsic parameters.

        Returns:
            A dict of the active camera's parameters.

            pose (numpy.ndarray): camera position in world coordinates,
            fov (float): horizontal field of view in radians
            focal_length (float)
            horizontal_aperture (float)
            view_projection_matrix (numpy.ndarray(dtype=float64, shape=(4, 4)))
            resolution (dict): resolution as a dict with 'width' and 'height'.
            clipping_range (tuple(float, float)): Near and Far clipping values.
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.editor.get_active_camera())
        prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        focal_length = prim.GetAttribute("focalLength").Get()
        horiz_aperture = prim.GetAttribute("horizontalAperture").Get()
        fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
        width = self.sd_interface.get_sensor_width(self.sd.SensorType.Rgb)
        height = self.sd_interface.get_sensor_height(self.sd.SensorType.Rgb)
        aspect_ratio = width / height
        near, far = prim.GetAttribute("clippingRange").Get()
        view_proj_mat = get_view_proj_mat(prim_tf, fov, aspect_ratio, near, far)

        return {
            "pose": np.array(prim_tf),
            "fov": fov,
            "focal_length": focal_length,
            "horizontal_aperture": horiz_aperture,
            "view_projection_matrix": view_proj_mat,
            "resolution": {"width": width, "height": height},
            "clipping_range": (near, far),
        }

    def get_pose(self):
        """Get pose of all objects with a semantic label.
        """
        stage = omni.usd.get_context().get_stage()
        mappings = self.get_instance_mappings()
        pose = {}
        for m in mappings:
            prim_path = m[0]
            prim = stage.GetPrimAtPath(prim_path)
            prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            pose[str(prim_path)] = prim_tf
        return pose

    def compute_3d_bounding_box_oobb(self):
        """Compute the 3D Object Oriented Bounding Box.

        Returns:
            numpy.ndarray(dtype=float64, shape=(N, 8, 3))
        """
        return np.array(self._get_bbox3d(self.sd.SensorType.BoundingBox3D))

    def _get_instance_mapping(self, cur_prim):
        instance_mappings = []
        descendant_instance_ids = []
        children = cur_prim.GetChildren()
        instance_id = self.sd_interface.get_instance_segmentation_id(str(cur_prim.GetPath())).tolist()

        for child in children:
            child_descendant_instance_ids, child_instance_mappings = self._get_instance_mapping(child)
            descendant_instance_ids += child_descendant_instance_ids
            instance_mappings += child_instance_mappings
        if instance_id:
            descendant_instance_ids += instance_id
        if cur_prim.HasAPI(Semantics.SemanticsAPI):
            semantic_label = cur_prim.GetAttribute("semantic:Semantics:params:semanticData").Get()
            semantic_id = self.sd_interface.get_semantic_segmentation_id_from_data(semantic_label)
            instance_mappings.append((str(cur_prim.GetPath()), semantic_id, semantic_label, descendant_instance_ids))
        return descendant_instance_ids, instance_mappings

    def get_instance_mappings(self):
        """Get mapping between prims with semantic labels and leaf instance IDs.
        Traverse through the scene graph and for each prim with a semantic label,
        store a list of the IDs of its leaf nodes.

        Returns:
            list of tuples mapping leaf instance IDs to a parent with a semantic label. Each tuple is represented by
            (<Path>, <Semantic ID>, <semantic Label>, <List of descendant instance IDs>). For
            example:

            [('/World/car', 1, 'Vehicle', [0, 1, 3, 4]), ('/World/car/tail_lights', 2, 'TailLights', [2, 3])]
        """
        stage = omni.usd.get_context().get_stage()
        _, self.instance_mappings = self._get_instance_mapping(stage.GetPseudoRoot())
        return self.instance_mappings

    def _reduce_bboxes(self, bboxes):
        """Reduce bounding boxes of leaf nodes to parents with a semantic label
        and add label to data.
        """
        instance_mappings = self.get_instance_mappings()
        reduced_bboxes = []
        for im in instance_mappings:
            if im[3]:  # if mapping has descendant instance ids
                mask = np.isin(bboxes["instanceId"], im[3])
                bbox_masked = bboxes[mask]
                if len(bbox_masked) > 0:
                    reduced_bboxes.append(
                        (
                            im[0],  # Prim path (name)
                            im[2],  # semanticLabel
                            im[1],  # semanticId
                            np.min(bbox_masked["x_min"]),
                            np.min(bbox_masked["y_min"]),
                            np.max(bbox_masked["x_max"]),
                            np.max(bbox_masked["y_max"]),
                        )
                    )

        return np.array(reduced_bboxes, dtype=[("name", "O"), ("semanticLabel", "O")] + bboxes.dtype.descr[1:])

    def _get_bbox2d(self, sensor):
        size = self.sd_interface.get_sensor_size(sensor)
        bboxes = self.sd_interface.get_sensor_host_bounding_box_2d_buffer_array(sensor, size)
        bboxes = self._reduce_bboxes(bboxes)
        return bboxes

    def _get_bbox3d(self, sensor):
        size = self.sd_interface.get_sensor_size(sensor)
        bboxes = self.sd_interface.get_sensor_host_bounding_box_3d_buffer_array(sensor, size)

        instance_mappings = self.get_instance_mappings()
        reduced_bboxes = []
        for im in instance_mappings:
            if im[3]:  # if mapping has descendant instance ids
                for box in bboxes:
                    if box[0] == im[3]:
                        local_range = Gf.Range3d(
                            Gf.Vec3d(box[2].astype(float), box[3].astype(float), box[4].astype(float)),
                            Gf.Vec3d(box[5].astype(float), box[6].astype(float), box[7].astype(float)),
                        )
                        bbox3d = Gf.BBox3d(local_range, Gf.Matrix4d(box[8].tolist()))
                        aligned_range = bbox3d.ComputeAlignedBox()
                        min_p = aligned_range.GetMin()
                        max_p = aligned_range.GetMax()

                        reduced_bboxes.append(
                            np.array(
                                [
                                    [min_p[0], min_p[1], min_p[2]],
                                    [min_p[0], min_p[1], max_p[2]],
                                    [min_p[0], max_p[1], min_p[2]],
                                    [min_p[0], max_p[1], max_p[2]],
                                    [max_p[0], min_p[1], min_p[2]],
                                    [max_p[0], min_p[1], max_p[2]],
                                    [max_p[0], max_p[1], min_p[2]],
                                    [max_p[0], max_p[1], max_p[2]],
                                ]
                            )
                        )

        return reduced_bboxes

    def get_groundtruth(self, gt_sensors):
        """Get groundtruth from specified gt_sensors.
        Enable syntheticdata sensors if required, render a frame and
        collect groundtruth from the specified gt_sensors

        If a sensor requiring RayTracedLighting mode is specified, render
        an additional frame in RayTracedLighting mode.

        Args:
            gt_sensors (list): List of strings of sensor names. Valid sensors names: rgb, depth,
                instanceSegmentation, semanticSegmentation, boundingBox2DTight,
                boundingBox2DLoose, boundingBox3D, camera

        Returns:
            Dict of sensor outputs
        """
        if isinstance(gt_sensors, str):
            gt_sensors = (gt_sensors,)

        rt_sensors = []
        remaining_sensors = []
        for sensor in gt_sensors:
            if sensor not in self.sensor_helpers:
                raise ValueError(
                    f"Sensor {sensor} is not supported. Choose from "
                    f"the following: {list(self.sensor_helpers.keys())}"
                )
            if sensor in RT_ONLY_SENSORS:
                rt_sensors.append(sensor)
            else:
                remaining_sensors.append(sensor)

        # Create/destroy sensors
        self.enable_sensors(gt_sensors)

        # Render frame, disabling this for perf reasons temporarily
        # self.app.update(0.0)

        gt = {}
        # Process non-RT-only sensors
        for sensor in remaining_sensors:
            gt[sensor] = self.sensor_helpers[sensor]()

        # Process only if we have sensors here
        if rt_sensors:
            # If using a sensor incompatible with current render mode, change mode and re-render
            cur_render_mode = self.carb_settings.get_as_string("/rtx/rendermode")
            if cur_render_mode != "RayTracedLighting":
                self.carb_settings.set_string("/rtx/rendermode", "RayTracedLighting")
                self.app.update(0.0)

            # Populate gt dict based on selected sensors
            for sensor in rt_sensors:
                gt[sensor] = self.sensor_helpers[sensor]()

            # Set render mode back to user-specified mode
            self.carb_settings.set_string("/rtx/rendermode", cur_render_mode)
        return gt


if __name__ == "__main__":
    # Example usage
    import random
    from omni.isaac.synthetic_utils import OmniKitHelper

    kit = OmniKitHelper()

    def add_semantic_label(prim, label):
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set(label)

    stage = kit.get_stage()
    for i in range(10):
        prim_type = random.choice(["Cube", "Sphere"])
        prim = stage.DefinePrim(f"/World/cube{i}", prim_type)
        translation = np.random.rand(3) * 300
        UsdGeom.XformCommonAPI(prim).SetTranslate(translation.tolist())
        UsdGeom.XformCommonAPI(prim).SetScale((50.0, 50.0, 50.0))
        add_semantic_label(prim, prim_type)

    sd_helper = SyntheticDataHelper()
    gt = sd_helper.get_groundtruth(
        [
            "rgb",
            "depth",
            "boundingBox2DTight",
            "boundingBox2DLoose",
            "instanceSegmentation",
            "semanticSegmentation",
            "boundingBox3D",
            "camera",
        ]
    )

    print(gt.keys())
