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
        self.mask = None
        self.partitions = []


    def make_partition(self, data):
        x = data.shape[0]
        y = data.shape[1]
        data_size = x * y
        indices = np.arange(data_size)
        np.random.shuffle(indices)
        partition_size = int(data_size * self.cd_rate)
        grouping = [indices[i:i+partition_size] for i in range(0, len(indices), partition_size)]

        for idxs_to_zero in grouping:
            tensor = data.clone()
            zero_indices_x = [idx // y for idx in idxs_to_zero]
            zero_indices_y = [idx % y for idx in idxs_to_zero]
            # Zero out the selected indices
            tensor[zero_indices_x, zero_indices_y] = 0
            tensor / (1 - (len(idxs_to_zero)/(data_size)) )
            self.partitions.append(tensor)



    def process_data(self, data):
        # Sample a new CD mask at each training step
        device = data.device
        # cd_mask = self.cd_input_dist.sample((data.shape[1],))
        cd_mask = self.cd_input_dist.sample(data.shape).to(device)
        # Store the grad_mask for later
        self.mask = cd_mask
        # Mask and scale post-CD input so it has the same sum as the original data
        # cd_masked_data = torch.where(cd_mask.bool(), data, torch.zeros_like(data))
        cd_masked_data = data * cd_mask.bool() / (1 - self.cd_rate)
        
        return cd_masked_data 
    

    def reset(self):
        self.mask = None
        self.partitions = []