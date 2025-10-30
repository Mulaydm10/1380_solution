import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any


def to_gpu(x):
    x = x.contiguous()
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def batch_to_gpu(batch):
    for key, value in batch.items():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = to_gpu(value)
    return batch


class Collate:

    def __call__(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates and processes a batch of data rows into a dictionary of tensors with unified
        dimensions, suitable for model input.

        Args:
            rows: A list of dictionaries, each representing a data item with various attributes
                  including images, maps, camera parameters, and bounding boxes.

        Returns:
            A dictionary containing processed and padded tensor batch data ready for model consumption.
        """
        result = {}
        all_keys = rows[0].keys()

        # Process simple tensor fields without padding
        for label in [
            "cond_cam",
            "camera_param",
            "height",
            "width",
        ]:
            if label in all_keys:
                values = torch.stack([torch.tensor(row[label]) for row in rows])
                result[label] = values.to(memory_format=torch.contiguous_format).float()

        # Process map data
        if "map" in all_keys:
            result["map"] = self._process_map_data(rows)

        # Process 3D bounding box data
        if "bboxes_3d_data" in all_keys:
            result["bboxes_3d_data"] = self._process_bbox_data(rows)

        # Process keys
        for label in ["ride_id", "camera_names"]:
            if label in all_keys:
                result[label] = [row[label] for row in rows]

        return result

    def _process_map_data(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process map-related data including centerlines, crosswalks, and traffic lights.

        Args:
            rows: List of input data samples

        Returns:
            Dictionary containing processed map data with tensors and valid masks
        """
        map_data = {}
        for label in ["centerlines", "crosswalks", "traffic_lights"]:
            torch_rows = []
            for row in rows:
                torch_rows.append(torch.from_numpy(row["map"][label]).transpose(1, 0))
            map_data[label] = (
                pad_sequence(
                    torch_rows,
                    batch_first=True,
                    padding_value=0,
                )
                .transpose(2, 1)
                .to(memory_format=torch.contiguous_format)
                .float()
            )

        return map_data

    def _process_bbox_data(self, rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process 3D bounding box data including boxes, masks, and classes.

        Args:
            rows: List of input data samples

        Returns:
            Dictionary containing processed bounding box data
        """

        bbox_data = {}
        for label in ["bboxes", "masks"]:
            torch_rows = [
                torch.tensor(row["bboxes_3d_data"][label]).transpose(2, 0) for row in rows
            ]
            pad_rows = pad_sequence(
                torch_rows,
                batch_first=True,
                padding_value=0,
            )
            bbox_data[label] = (
                pad_rows.transpose(3, 1).to(memory_format=torch.contiguous_format).float()
            )

        torch_rows = [
            torch.tensor(row["bboxes_3d_data"]["classes"]).transpose(2, 0) for row in rows
        ]
        pad_rows = pad_sequence(
            torch_rows,
            batch_first=True,
            padding_value=-1,
        )
        bbox_data["classes"] = (
            pad_rows.transpose(3, 1).to(memory_format=torch.contiguous_format).float()
        )
        return bbox_data
