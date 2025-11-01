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


import torch
import torch
import numpy as np

def pad_collate(batch):
    """
    A collate function that pads variable-length tensors to the max length in a batch.
    It also creates an attention mask for the padded elements.
    """
    # Separate keys that need padding from those that don't
    variable_len_keys = ["bboxes", "classes", "masks"]
    result = {}

    # Handle standard collation for most keys
    for key in batch[0].keys():
        if key != "bboxes_3d_data":
            result[key] = torch.utils.data.default_collate([d[key] for d in batch])

    # --- Custom Padding for bboxes_3d_data ---
    bbox_data = [d["bboxes_3d_data"] for d in batch]
    result["bboxes_3d_data"] = {}

    # Find the max number of objects in this batch
    max_num_objects = max(d["bboxes"].shape[2] for d in bbox_data)

    for key in variable_len_keys:
        padded_tensors = []
        for i in range(len(batch)):
            tensor = bbox_data[i][key]
            num_objects = tensor.shape[2]
            pad_size = max_num_objects - num_objects
            
            # The shape is (1, 1, num_objects, ...), so padding is on the 3rd dimension (index 2)
            # We need to define padding for all dimensions after the one we are padding
            # (pad_left, pad_right, pad_top, pad_bottom, ...)
            padding = (0, 0) * (tensor.ndim - 3) + (0, pad_size)
            padded_tensor = np.pad(tensor, ((0,0), (0,0), (0, pad_size)) + ((0,0),)*(tensor.ndim-3), 'constant', constant_values=0)
            padded_tensors.append(padded_tensor)
        
        result["bboxes_3d_data"][key] = torch.from_numpy(np.concatenate(padded_tensors, axis=0))

    # Create an attention mask for the bounding boxes
    # Shape: (B, 1, 1, max_num_objects)
    attention_masks = []
    for i in range(len(batch)):
        num_objects = bbox_data[i]["bboxes"].shape[2]
        mask = np.zeros((1, 1, max_num_objects), dtype=np.float32)
        mask[:, :, :num_objects] = 1.0
        attention_masks.append(mask)
    
    result["bboxes_3d_data"]["attention_mask"] = torch.from_numpy(np.concatenate(attention_masks, axis=0))

    return result

class Collate:
    def __call__(self, batch):
        return pad_collate(batch)
