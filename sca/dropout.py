import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

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
        # Use FIFO for grad masks
        self.mask = None

    def process_data(self, data):
        # Sample a new CD mask at each training step
        device = data.device
        cd_mask = self.cd_input_dist.sample((data.shape[1],))
        # Store the grad_mask for later
        self.mask = cd_mask
        # Mask and scale post-CD input so it has the same sum as the original data
        # cd_masked_data = torch.where(cd_mask.bool(), data, torch.zeros_like(data))
        cd_masked_data = data * cd_mask.bool() / (1 - self.cd_rate)
        
        return cd_masked_data 
    
    # def process_data(self, data):
    #     # Sample a new CD mask at each training step
    #     device = data.device
    #     cd_mask = self.cd_input_dist.sample(data.shape).to(device)
    #     # Store the grad_mask for later
    #     self.mask = cd_mask
    #     # Mask and scale post-CD input so it has the same sum as the original data
    #     cd_masked_data = data * cd_mask / (1 - self.cd_rate)
    #     cd_input = torch.tensor(cd_masked_data)

    #     return cd_input 


    # def process_losses(self, recon_loss, *args):
    #     # First-in-first-out
    #     grad_mask = self.grad_masks.pop(0)
    #     # Expand mask, but don't block gradients
    #     # grad_mask = pad_mask(grad_mask, recon_loss, 1.0) # I don't think this will be necessary
    #     # Block gradients with respect to the masked outputs
    #     grad_loss = recon_loss * grad_mask
    #     nograd_loss = (recon_loss * (1 - grad_mask)).detach()
    #     cd_loss = grad_loss + nograd_loss

    #     return cd_loss

    def reset(self):
        self.mask = None