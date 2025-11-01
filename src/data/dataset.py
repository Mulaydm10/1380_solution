import numpy as np
import torch
from torch.utils.data import Dataset
from .types import Scene
from .bev_raster import rasterize_bev
from PIL import Image
import torchvision.transforms as T
import logging
import os

logger = logging.getLogger(__name__)

class SensorGenDataset(Dataset):
    def __init__(self, scenes, data_root, mode='train', num_cond_cams=4, image_size=(512, 512)):
        """
        Аргументы:
            scenes (list): путь до файлов
        """
        self.scenes = scenes
        self.data_root = data_root
        self.mode = mode
        self.image_size = image_size

        self.target_cameras = [
            "camera_camera_inner_frontal_middle", "camera_rear",
            "camera_side_left_backward", "camera_side_left_forward", "camera_side_right_forward"
        ]
        # Define a flexible pool of conditioning cameras
        cond_camera_pool = self.target_cameras[:3] + ["camera_side_right_backward"]
        self.cond_cameras = cond_camera_pool[:num_cond_cams] # Select N cameras from the pool

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # To [-1,1]
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        result = {}
        scene_path = self.data_root / self.scenes[idx]
        scene = Scene.from_path(scene_path)

        # bbox preprocessing (as before)
        result["bboxes_3d_data"] = {}
        result["bboxes_3d_data"]["bboxes"] = scene.bboxes.bboxes_np[None, None]
        result["bboxes_3d_data"]["classes"] = scene.bboxes.classes_np[None, None]
        result["bboxes_3d_data"]["masks"] = np.array(
            [scene.bboxes.per_camera_visibility[label] for label in scene.camera_names]
        )[None]

        # camera params preprocessing (as before)
        intrinsics = np.array(
            [scene.camera_params.get_intrinsics_matrix(label) for label in scene.camera_names]
        )[None]
        extrinsics = np.array(
            [scene.camera_params.get_extrinsics_matrix(label) for label in scene.camera_names]
        )[None]
        result["camera_param"] = np.concatenate([intrinsics, extrinsics[:, :, :-1]], axis=-1)

        # --- NEW: Load Ground Truth Images (only in train mode) ---
        if self.mode == 'train':
            gt_images = []
            for cam_name in self.target_cameras:
                img_path = os.path.join(scene_path, 'images_gt', f"{cam_name}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    gt_images.append(self.transform(img))
                else:
                    logger.warning(f"Missing GT image: {img_path}")
                    gt_images.append(torch.zeros(3, *self.image_size))
            result['images_gt'] = torch.stack(gt_images)

        # --- NEW: Load Conditioning Images for Training Mode ---
        if self.mode == 'train':
            cond_images = []
            for cam_name in self.cond_cameras:
                # Handle the one image that is in a different directory
                if cam_name == "camera_side_right_backward":
                    subdir = 'images_input'
                else:
                    subdir = 'images_gt'
                img_path = os.path.join(scene_path, subdir, f"{cam_name}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    cond_images.append(self.transform(img))
                else:
                    logger.warning(f"Missing conditioning image: {img_path}")
                    cond_images.append(torch.zeros(3, *self.image_size))
            result['cond_cam_raw'] = torch.stack(cond_images)
        else: # Inference mode
            # Load single conditioning image as before
            img_path = os.path.join(scene_path, 'images_input', "camera_side_right_backward.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                result['cond_cam_raw'] = self.transform(img).unsqueeze(0) # Add batch dim for consistency
            else:
                result['cond_cam_raw'] = torch.zeros(1, 3, *self.image_size)

        # map preprocessing (as before)
        result["map"] = {}
        if scene.map_data.centerlines.data:
            result["map"]["centerlines"] = np.array(scene.map_data.centerlines.data)[None]
        else:
            result["map"]["centerlines"] = np.zeros([1, 0, 6, 10])

        if scene.map_data.crosswalks.data:
            result["map"]["crosswalks"] = np.array(scene.map_data.crosswalks.data)[None]
        else:
            result["map"]["crosswalks"] = np.zeros([1, 0, 8, 4])

        if scene.map_data.traffic_lights.data:
            result["map"]["traffic_lights"] = np.array(scene.map_data.traffic_lights.data)[None]
        else:
            result["map"]["traffic_lights"] = np.zeros([1, 0, 13])

        result["height"] = self.image_size[0]
        result["width"] = self.image_size[1]

        result["ride_id"] = str(scene.ride_id)
        result["camera_names"] = scene.camera_names

        if self.mode == 'train':
            result['bev_grid'] = rasterize_bev(result['bboxes_3d_data'], result['map'])

        return result
