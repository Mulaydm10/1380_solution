import numpy as np
import torch
from torch.utils.data import default_collate

def general_ragged_pad(arr_list, pad_val=0.0):
    """General pad for ragged np arrays (var in any dim); return {'data': stacked_padded, 'mask': stacked_bool [B,*max_shape]}."""
    if not arr_list or not isinstance(arr_list[0], np.ndarray) or len(arr_list[0].shape) < 2:  # Skip scalar/1D
        return default_collate([torch.as_tensor(a) for a in arr_list])
    
    shapes = [a.shape for a in arr_list]
    ndim = len(shapes[0])
    max_shape = tuple(max(s[i] for s in shapes) for i in range(ndim))
    
    padded = []
    masks = []  # Full max_shape bool per arr
    for idx, arr in enumerate(arr_list):
        # Pad data: Per-dim post-pad
        pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(ndim)]
        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=pad_val)
        padded.append(padded_arr)
        
        # Mask: Zeros max_shape; True on original shape
        mask = np.zeros(max_shape, dtype=bool)
        slices = tuple(slice(0, arr.shape[i]) for i in range(ndim))
        mask[slices] = True
        masks.append(mask)
    
    padded_data = torch.from_numpy(np.stack(padded)).float()
    mask_data = torch.from_numpy(np.stack(masks)).bool()  # [B, *max_shape] – Squeeze later if attn needs [B,seq]
    
    return {'data': padded_data, 'mask': mask_data}

def pad_collate_recursive(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return default_collate(batch)
    elif isinstance(elem, np.ndarray):
        return general_ragged_pad(batch)  # New: Catch inner np var-len
    elif isinstance(elem, (list, tuple)):
        # Check if any item in the sublists is a string
        try:
            # This will fail if there are strings
            np.array(batch, dtype=np.float32)
            is_numeric = True
        except (TypeError, ValueError):
            is_numeric = False

        if not is_numeric:
            # Pad string lists with ''
            lengths = [len(d) for d in batch]
            max_len = max(lengths)
            padded_lists = []
            for d in batch:
                padded = d + [''] * (max_len - len(d))  # List pad
                padded_lists.append(padded)
            return padded_lists  # Return list of lists, no tensor conversion
        else:
            # Numeric pad + mask
            lengths = [len(d) for d in batch]
            max_len = max(lengths)
            padded_data = []
            masks = []
            for d in batch:
                arr_d = np.array(d, dtype=np.float32)  # Ensure numeric
                pad_len = max_len - len(arr_d)
                padded = np.pad(arr_d, ((0, pad_len),) + ((0,0),)*(arr_d.ndim-1), mode='constant', constant_values=0)
                padded_data.append(padded)
                
                mask = np.ones(len(arr_d), dtype=bool)
                mask = np.pad(mask, (0, pad_len), mode='constant', constant_values=False)
                masks.append(mask)
            
            return {'data': torch.from_numpy(np.stack(padded_data)), 'mask': torch.from_numpy(np.stack(masks))}

    elif isinstance(elem, dict):
        result = {}
        for key in elem:
            collated_value = pad_collate_recursive([d[key] for d in batch])
            if isinstance(collated_value, dict) and 'data' in collated_value:
                result[key] = collated_value  # {'data': tensor, 'mask': tensor}
            else:
                result[key] = collated_value
        return result
    else:
        return default_collate(batch)

class Collate:
    def __call__(self, batch):
        # More robust handling of bboxes_3d_data
        print("--- Custom Collate Start ---")
        
        # Separate bboxes from the rest of the batch
        bboxes_list = [d.pop('bboxes_3d_data') for d in batch]
        
        # Collate the rest of the batch using the recursive function
        collated_batch = pad_collate_recursive(batch)
        print(f"[Collate] Collated batch keys: {list(collated_batch.keys())}")

        # Now, specifically collate the bboxes
        # We need to extract the actual numpy arrays from the nested dict
        bbox_arrays = [b['bboxes'] for b in bboxes_list]
        collated_bboxes = general_ragged_pad(bbox_arrays)
        print(f"[Collate] Collated bboxes['data'] shape: {collated_bboxes['data'].shape}")
        print(f"[Collate] Collated bboxes['mask'] shape: {collated_bboxes['mask'].shape}")

        # Re-integrate the correctly padded bboxes into the final batch
        collated_batch['bboxes_3d_data'] = {
            'bboxes': collated_bboxes
        }
        
        print("--- Custom Collate End ---")
        return collated_batch