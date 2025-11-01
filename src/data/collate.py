import torch
import numpy as np

def pad_tensor(data, max_len):
    """Pads a list of numpy arrays to a max length and creates a mask."""
    padded_data = []
    masks = []
    for item in data:
        # This assumes the variable dimension is the first one.
        pad_width = [(0, max_len - item.shape[0])] + [(0, 0)] * (item.ndim - 1)
        padded_item = np.pad(item, pad_width, 'constant', constant_values=0)
        padded_data.append(padded_item)

        mask = np.zeros(max_len, dtype=np.float32)
        mask[:item.shape[0]] = 1.0
        masks.append(mask)
    
    return torch.from_numpy(np.stack(padded_data)), torch.from_numpy(np.stack(masks))

def pad_collate_recursive(batch):
    """
    A recursive collate function that pads variable-length tensors in a batch.
    """
    elem = batch[0]
    if isinstance(elem, np.ndarray):
        # Check if this is a variable-length item by comparing shapes
        if any(b.shape[0] != elem.shape[0] for b in batch):
            max_len = max(b.shape[0] for b in batch)
            padded_data, masks = pad_tensor(batch, max_len)
            return {"data": padded_data, "mask": masks}
        else:
            return torch.utils.data.default_collate(batch)

    elif isinstance(elem, dict):
        result = {}
        for key in elem:
            collated_value = pad_collate_recursive([d[key] for d in batch])
            # If padding returned a dict with data and mask, unpack it
            if isinstance(collated_value, dict) and 'data' in collated_value:
                result[key] = collated_value['data']
                result[f"{key}_mask"] = collated_value['mask']
            else:
                result[key] = collated_value
        return result

    # Fallback for other data types (like strings, fixed-size tensors)
    return torch.utils.data.default_collate(batch)

class Collate:
    def __call__(self, batch):
        # The recursive function will handle the nested structure automatically
        return pad_collate_recursive(batch)