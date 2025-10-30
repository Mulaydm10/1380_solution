import numpy as np
from torch.utils.data import Dataset
from .types import Scene


class SensorGenDataset(Dataset):
    def __init__(self, scenes, data_root):
        """
        Аргументы:
            scenes (list): путь до файлов
        """
        self.scenes = scenes
        self.data_root = data_root

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        result = {}
        scene = Scene.from_path(self.data_root / self.scenes[idx])

        # bbox preprocessing
        result["bboxes_3d_data"] = {}
        result["bboxes_3d_data"]["bboxes"] = scene.bboxes.bboxes_np[None, None]
        result["bboxes_3d_data"]["classes"] = scene.bboxes.classes_np[None, None]
        result["bboxes_3d_data"]["masks"] = np.array(
            [scene.bboxes.per_camera_visibility[label] for label in scene.camera_names]
        )[None]

        # camera params preprocessing
        intrinsics = np.array(
            [scene.camera_params.get_intrinsics_matrix(label) for label in scene.camera_names]
        )[None]
        extrinsics = np.array(
            [scene.camera_params.get_extrinsics_matrix(label) for label in scene.camera_names]
        )[None]
        result["camera_param"] = np.concatenate([intrinsics, extrinsics[:, :, :-1]], axis=-1)

        # camera images preprocessing
        cond_cam = np.array(scene.get_image("/side/right/backward"))
        cond_cam = cond_cam.transpose(2, 0, 1)[None, None]
        result["cond_cam"] = (cond_cam / 127.5) - 1.0

        # map preprocessing
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

        result["height"] = result["cond_cam"].shape[3]
        result["width"] = result["cond_cam"].shape[4]

        result["ride_id"] = str(scene.ride_id)
        result["camera_names"] = scene.camera_names

        return result
