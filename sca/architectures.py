import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

import geotorch
import numpy as np



# Note - I would like to credit the pytorch tutorial, that the formatting of my functions is similar to:
# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

class LowROrth(nn.Module):
    """
    Class for SCA model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowROrth, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b
        geotorch.orthogonal(self.fc2,"weight") #Make V orthogonal

    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output



class Sphere(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)
    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)


class LowRNorm(nn.Module):
    """
    Class for SCA (with unit norm, but not orthogonal) model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowRNorm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias

        self.fc2 = P.register_parametrization(nn.Linear(self.hidden_size, self.output_size), "weight", Sphere(dim=0))
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b

        self._x_vals = torch.arange(-10,11,1)
        
        # TODO: learned sigmas, currently we give it a default

        # self.sigmas = torch.nn.Parameter(torch.ones((
        #     n_regions,
        #     self.n_components
        # )))
    
    def _gauss(self, x, mu, sig):
        return torch.exp(-0.5*((x-mu)/sig)**2)
    
    def _make_filters(self):
        """
        Creates Gaussian filters for each latent dimension
        """
        filt_bank = [ 
            self._gauss(
                self._x_vals,
                0,
                # stddev[j]).reshape(1,1,-1)
                1).reshape(1,1,-1)
            for j in range(self.hidden_size)
        ]
        return filt_bank
    
    def apply_filters(self, f, z):
        return torch.stack(
            [F.conv1d(
                z[:,j].reshape(1,1,-1), # should force z to have 3 dimensions
                filt # we want this to have 3 or 4 dimensions  f should be (n,1,1,20)
            ) for j, filt in enumerate(f)
            ]).squeeze().T
        

    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """
        z_pre_filter = self.fc1(x)
        
        f = self._make_filters()
        z_post_filter = self.apply_filters(f, z_pre_filter)
        X_hat = self.fc2(z_post_filter)
        # X_hat = torch.nn.functional.relu(X_hat)
        X_hat_pos = torch.nn.functional.softplus(X_hat)
        return z_pre_filter, X_hat_pos


        # after we get the convolutions to work, we will need to trim target size in loss function so target and prediction sizes are the same
    
        # adding parameter back into the function so it can be learned by the model