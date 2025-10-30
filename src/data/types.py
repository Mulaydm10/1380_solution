import dataclasses
import json
import pathlib

from typing import ClassVar, Dict, List

import cv2
import numpy as np
import scipy.spatial.transform


CAMERA_INPUT = "/side/right/backward"
CAMERAS_OTHER = [
    "/side/left/forward",
    "/camera/inner/frontal/middle"
    "/side/right/forward",
    "/rear",
    "/side/left/backward",
]


def sanitize_camera_name(name: str) -> str:
    # e.g. /side/left/forward -> side_left_forward
    return name.strip("/").replace("/", "_").replace(" ", "_").lower()


def read_image_cv2(image_path: pathlib.Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image from path {image_path}.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclasses.dataclass
class RideID:
    ride_date: str
    ride_time: str
    log_time: str
    rover: str
    message_ts: int  # nanoseconds


# It would be much better to store bbox as [extent, qvec, tvec], but for simplicity we store 8 corners.
BBox = List[List[float]]  # 8x3 bounding box


@dataclasses.dataclass
class BBoxesData:
    """Bounding boxes data structure. Each bounding box represents single object in 3D space with 8 corner points."""
    bboxes: List[BBox]  # [N_boxes, 8, 3]
    classes: List[int]  # [N_boxes]
    per_camera_visibility: Dict[str, List[int]]  #  camera label -> [N_boxes], 1 if box visible in camera, 0 otherwise

    @property
    def bboxes_np(self) -> np.ndarray:
        return np.array(self.bboxes, dtype=np.float32)

    @property
    def classes_np(self) -> np.ndarray:
        return np.array(self.classes, dtype=np.int32)

    def __post_init__(self):
        n_boxes = len(self.bboxes)
        if len(self.bboxes) != len(self.classes):
            raise ValueError("Number of bounding boxes must match number of classes.")
        for mask in self.per_camera_visibility.values():
            if len(mask) != n_boxes:
                raise ValueError("Each mask must have the same number of elements as there are bounding boxes.")

        for bbox in self.bboxes:
            if len(bbox) != 8:
                raise ValueError("Each bounding box must have 8 corner points.")
            for corner in bbox:
                if len(corner) != 3:
                    raise ValueError("Each corner point must have 3 coordinates (x, y, z).")


@dataclasses.dataclass
class MapCenterlinesData:
    """
    Centerlines data structure. Each centerline consists of 6 segments, and each segment is represented by 10 features.
    Features per segment:
    - x_start: float - X coordinate of the start point of the segment
    - y_start: float - Y coordinate of the start point of the segment
    - x_end: float - X coordinate of the end point of the segment
    - y_end: float - Y coordinate of the end point of the segment
    - width: float - Width of the lane at this segment
    - lane_type: - ???
    - left_markline: - ???
    - right_markline: - ???
    - start_z_level: float
    - finish_z_level: float
    """

    data: List[List[List[float]]]  # [N_centerlines, N_segments, N_features]

    _N_SEGMENTS: ClassVar[int] = 6
    _N_FEATURES: ClassVar[int] = 10

    @property
    def n_centerlines(self) -> int:
        return len(self.data)

    def __post_init__(self):
        for centerline in self.data:
            if len(centerline) != self._N_SEGMENTS:
                raise ValueError(f"Each centerline must have {self._N_SEGMENTS} segments.")
            for segment in centerline:
                if len(segment) != self._N_FEATURES:
                    raise ValueError(f"Each segment must have {self._N_FEATURES} features.")


@dataclasses.dataclass
class MapCrosswalksData:
    """
    Crosswalks data structure. Each crosswalk consists of 8 segments, and each segment is represented by 4 features.
    Features per segment:
    - x_start: float - X coordinate of the start point of the segment
    - y_start: float - Y coordinate of the start point of the segment
    - x_end: float - X coordinate of the end point of the segment
    - y_end: float - Y coordinate of the end point of the segment
    """

    data: List[List[List[float]]]  # [N_crosswalks, N_segments, N_features]

    _N_SEGMENTS: ClassVar[int] = 8
    _N_FEATURES: ClassVar[int] = 4

    @property
    def n_crosswalks(self) -> int:
        return len(self.data)

    def __post_init__(self):
        for crosswalk in self.data:
            if len(crosswalk) != self._N_SEGMENTS:
                raise ValueError(f"Each crosswalk must have {self._N_SEGMENTS} segments.")
            for segment in crosswalk:
                if len(segment) != self._N_FEATURES:
                    raise ValueError(f"Each segment must have {self._N_FEATURES} features.")


@dataclasses.dataclass
class MapTrafficLightsData:
    """
    Traffic lights data structure. Each traffic light consists of 1 segment, and each segment is represented by 13 features.
    Features per segment:
    - location_xyz: List[float] - [x, y, z] coordinates of the traffic light location
    - direction_vector_xyz: List[float] - [dx, dy, dz] direction vector of the traffic light
    - traffic_light_type: int - Type of the traffic light
    - sections: List[bool] - [has_main_section, has_left_section, has_right_section]
    - sections_value: List[int] - ???
    """

    data: List[List[float]]  # [N_traffic_lights, N_features]

    _N_SEGMENTS: ClassVar[int] = 1
    _N_FEATURES: ClassVar[int] = 13

    @property
    def n_traffic_lights(self) -> int:
        return len(self.data)

    def __post_init__(self):
        for traffic_light in self.data:
            if len(traffic_light) != self._N_FEATURES:
                raise ValueError(f"Each traffic light must have {self._N_FEATURES} features.")


@dataclasses.dataclass
class MapData:
    centerlines: MapCenterlinesData
    crosswalks: MapCrosswalksData
    traffic_lights: MapTrafficLightsData

    @property
    def centerlines_np(self) -> np.ndarray:
        return np.array(self.centerlines.data, dtype=np.float32)

    @property
    def crosswalks_np(self) -> np.ndarray:
        return np.array(self.crosswalks.data, dtype=np.float32)

    @property
    def traffic_lights_np(self) -> np.ndarray:
        return np.array(self.traffic_lights.data, dtype=np.float32)


@dataclasses.dataclass
class CameraParams:
    """Camera parameters data structure.  Extrinsics are represented as quaternion + translation vector."""
    intrinsics: List[float]  # [fx, fy, cx, cy]
    qvec: List[float]  # [qw, qx, qy, qz]
    tvec: List[float]  # [tx, ty, tz]

    @property
    def intrinsics_matrix(self) -> np.ndarray:
        """Returns the camera intrinsics matrix."""
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    @property
    def extrinsics_matrix(self) -> np.ndarray:
        """Returns the camera extrinsics matrix (4x4) as a transformation from camera to world coordinates."""
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = scipy.spatial.transform.Rotation.from_quat(self.qvec, scalar_first=True).as_matrix()
        extrinsics[:3, 3:] = np.array(self.tvec, dtype=np.float32).reshape(3, 1)
        return extrinsics


@dataclasses.dataclass
class CameraParamsData:
    camera_params: Dict[str, CameraParams]  # camera label -> CameraParams

    def get_intrinsics_matrix(self, camera_id: str) -> np.ndarray:
        if camera_id not in self.camera_params:
            raise ValueError(f"Camera ID {camera_id} not found in camera parameters.")
        return self.camera_params[camera_id].intrinsics_matrix

    def get_extrinsics_matrix(self, camera_id: str) -> np.ndarray:
        if camera_id not in self.camera_params:
            raise ValueError(f"Camera ID {camera_id} not found in camera parameters.")
        return self.camera_params[camera_id].extrinsics_matrix


@dataclasses.dataclass
class Scene:
    """
    Scene example structure:
    ➜  ml_cup_test_data tree ride_20250606_120023_130207_96b8f576c788b75e_1749204719706332293
    ride_20250606_120023_130207_96b8f576c788b75e_1749204719706332293
    ├── bboxes_3d_data.json
    ├── camera_params.json
    ├── images_input
    |   ├── camera_side_right_backward.jpg
    ├── images_gt
    │   ├── camera_camera_inner_frontal_middle.jpg
    │   ├── camera_rear.jpg
    │   ├── camera_side_left_backward.jpg
    │   ├── camera_side_left_forward.jpg
    │   └── camera_side_right_forward.jpg
    ├── map_data.json
    └── ride_id.json

    1 directory, 10 files

    JSON files were saved using dataclasses.asdict(), e.g.:
    ride_id: RideId = ...
    ride_id_json = json.dumps(dataclasses.asdict(ride_id), indent=2).encode("utf-8")
    """
    ride_id: RideID
    camera_names: List[str]
    image_paths: Dict[str, pathlib.Path]
    pred_image_paths: Dict[str, pathlib.Path]
    camera_params: CameraParamsData
    bboxes: BBoxesData
    map_data: MapData

    def get_image(self, camera_id: str) -> np.ndarray:
        """Reads and returns the image for the given camera ID."""
        image_path = self.image_paths.get(camera_id)
        if image_path is None:
            raise ValueError(f"Camera ID {camera_id} not found in image paths.")
        return read_image_cv2(image_path)

    @classmethod
    def from_path(cls, scene_path: pathlib.Path) -> "Scene":

        def _load_json(file_path: pathlib.Path):
            return json.loads(file_path.read_text())

        ride_id: RideID = RideID(**_load_json(scene_path / "ride_id.json"))

        camera_params_dict = _load_json(scene_path / "camera_params.json")
        camera_params_data = CameraParamsData(
            camera_params={k: CameraParams(**v) for k, v in camera_params_dict["camera_params"].items()}
        )
        bboxes_data = BBoxesData(**_load_json(scene_path / "bboxes_3d_data.json"))

        map_data_dict = _load_json(scene_path / "map_data.json")
        map_data = MapData(
            centerlines=MapCenterlinesData(**map_data_dict["centerlines"]),
            crosswalks=MapCrosswalksData(**map_data_dict["crosswalks"]),
            traffic_lights=MapTrafficLightsData(**map_data_dict["traffic_lights"]),
        )

        camera_names = list(camera_params_data.camera_params.keys())

        def _compose_image_path(camera_name: str, images_dir: str) -> pathlib.Path:
            return scene_path / images_dir / f"camera_{sanitize_camera_name(camera_name)}.jpg"

        def _collect_images(images_dir: str) -> Dict[str, pathlib.Path]:
            image_paths = {
                camera_name: scene_path / images_dir / f"camera_{sanitize_camera_name(camera_name)}.jpg"
                for camera_name in CAMERAS_OTHER
            }
            image_paths[CAMERA_INPUT] = _compose_image_path(CAMERA_INPUT, "images_input")
            image_paths = {k: v for k, v in image_paths.items() if v.exists()}
            return image_paths

        image_paths = _collect_images("images_gt")
        pred_image_paths = _collect_images("images_pred")

        return cls(
            ride_id=ride_id,
            camera_names=camera_names,
            image_paths=image_paths,
            pred_image_paths=pred_image_paths,
            camera_params=camera_params_data,
            bboxes=bboxes_data,
            map_data=map_data,
        )
