import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

class CoordinatedDropout:
    def __init__(self, cd_rate=.25):
        self.cd_rate = cd_rate
        self.cd_input_dist = Bernoulli(1 - cd_rate)
        # Use FIFO for grad masks
        self.mask = None

    def process_data(self, data):
        # Sample a new CD mask at each training step
        
        cd_mask = self.cd_input_dist.sample(data.shape)
        # Store the grad_mask for later
        self.mask = cd_mask
        # Mask and scale post-CD input so it has the same sum as the original data
        cd_masked_data = data * cd_mask.bool() / (1 - self.cd_rate)
        
        return cd_masked_data 

    def reset(self):
        self.mask = None