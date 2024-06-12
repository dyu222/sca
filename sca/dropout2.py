import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np

# def pad_mask(mask, data, value):
#     """Adds padding to I/O masks for CD and SV in cases where
#     reconstructed data is not the same shape as the input data.
#     """
#     t_forward = data.shape[1] - mask.shape[1]
#     n_heldout = data.shape[2] - mask.shape[2]
#     pad_shape = (0, n_heldout, 0, t_forward)
#     return F.pad(mask, pad_shape, value=value)

class CoordinatedDropout:
    def __init__(self, cd_rate=.25):
        self.cd_rate = cd_rate
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        self.partitions = []


    def make_partition(self, rows, cols):
        data_size = rows * cols
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        partition_size = int(data_size * self.cd_rate)
        grouping = [indices[i:i+partition_size] for i in range(0, len(indices), partition_size)]

        for idxs_to_zero in grouping:
            # tensor = data.clone()       # probably what is making this take a long time. could just make the masks and multiply it by the data like in the other one
            tensor = torch.ones((rows,cols))
            zero_indices_x = [idx // cols for idx in idxs_to_zero]
            zero_indices_y = [idx % cols for idx in idxs_to_zero]
            # Zero out the selected indices
            tensor[zero_indices_x, zero_indices_y] = 0
            # tensor / (1 - (len(idxs_to_zero)/(data_size)) )
            dropout_percent_complement = 1 - (len(idxs_to_zero)/(data_size))
            self.partitions.append((tensor,dropout_percent_complement))

    def reset(self):
        self.partitions = []