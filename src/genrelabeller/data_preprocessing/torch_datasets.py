"""Module to create the torch datasets for model training."""

import numpy as np
import torch


class MusicDataset(torch.utils.data.Dataset):
    """Class to create the torch datasets for model training."""

    def __init__(
        self,
        input_data: np.ndarray,
        target_data: np.ndarray = None,
    ):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_arrays = np.array(self.input_data[idx])
        if self.target_data is not None:
            target_arrays = np.array(self.target_data[idx])
            self.target_tensors = torch.tensor(target_arrays, dtype=torch.float32)
        else:
            self.target_tensors = None
        self.input_tensors = torch.tensor(input_arrays, dtype=torch.float32)
        return self.input_tensors, self.target_tensors
