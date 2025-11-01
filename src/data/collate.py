import numpy as np
import torch
from torch.utils.data import default_collate

def pad_nested_var_len(arr_list, pad_dim=2, pad_val=0.0):
    """Pad multi-dimensional np arrays (e.g., 5D) along a specified variable dimension."""
    if not arr_list or not isinstance(arr_list[0], np.ndarray) or len(arr_list[0].shape) < 3:
        return default_collate([torch.as_tensor(a) for a in arr_list])
    
    shapes = [a.shape for a in arr_list]
    max_n = max(s[pad_dim] for s in shapes)
    padded = []
    masks = []
    
    for arr in arr_list:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[pad_dim] = (0, max_n - arr.shape[pad_dim])
        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_val)
        padded.append(padded_arr)
        
        # Create a mask for the valid (non-padded) part
        mask = np.zeros(max_n, dtype=bool)
        mask[:arr.shape[pad_dim]] = True
        masks.append(mask)

    padded_data = torch.from_numpy(np.stack(padded)).float()
    mask_data = torch.from_numpy(np.stack(masks)).bool()
    
    return {'data': padded_data, 'mask': mask_data}

def pad_collate_recursive(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return default_collate(batch)
    elif isinstance(elem, np.ndarray):
        # This will now handle the nested numpy arrays like bboxes
        return pad_nested_var_len(batch)
    elif isinstance(elem, dict):
        result = {}
        for key in elem:
            collated_value = pad_collate_recursive([d[key] for d in batch])
            if isinstance(collated_value, dict) and 'data' in collated_value:
                # Unpack the data and mask from the returned dictionary
                result[key] = collated_value['data']
                result[f"{key}_mask"] = collated_value['mask']
            else:
                result[key] = collated_value
        return result
    else:
        # Fallback for strings, numbers, etc.
        return default_collate(batch)

class Collate:
    def __call__(self, batch):
        return pad_collate_recursive(batch)