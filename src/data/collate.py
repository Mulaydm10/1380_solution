import torch
from torch.nn.utils.rnn import pad_sequence

class Collate:
    def __call__(self, batch):
        result = {}
        # List of keys that are known to have variable lengths
        # We will pad these, and collate others normally.
        variable_len_keys = ["bboxes_3d", "map_lanes", "traffic_lights", "map_crosswalks"] # Add any other variable keys here

        # First, collate all keys that do not need special handling
        for key in batch[0].keys():
            if key not in variable_len_keys:
                try:
                    items = [d[key] for d in batch]
                    result[key] = torch.stack(items)
                except Exception as e:
                    # Handle non-tensor data like ride_id, camera_names
                    result[key] = [d[key] for d in batch]

        # Now, handle variable-length keys with padding
        for key in variable_len_keys:
            if key in batch[0]:
                items = [torch.from_numpy(d[key]) for d in batch]
                # pad_sequence expects a list of tensors
                # We assume the variable dimension is the first one (num_objects)
                padded_items = pad_sequence(items, batch_first=True, padding_value=0.0)
                result[key] = padded_items

        return result

