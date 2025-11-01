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
from torch.utils.data.dataloader import default_collate

class Collate:
    def __call__(self, batch):
        # First, use the default collate to handle most data types
        collated_batch = default_collate(batch)

        # The default collate will turn our list of stacked tensors into a single tensor,
        # which is what we want. No special handling is needed if __getitem__ returns stacked tensors.
        # This function remains for clarity and future modifications.

        return collated_batch
