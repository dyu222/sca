import torch
from torch.special import gammaln
import math

def my_loss(output, target, latent, lam_sparse, sample_weight):

    """
    Loss function

    Parameters
    ----------
    output: the predictions
        torch 2d tensor of size [n_time, output_size]
    target: ground truth output
        torch 2d tensor of size [n_time, output_size]
    latent: low dimensional representations
        torch 2d tensor of size [n_time, hidden_size]
    lam_sparse: sparsity penalty weight
        scalar
    sample_weight: weighting of each sample
        torch 2d tensor of size [n_time, 1]


    Returns
    -------
    loss: the value of the cost function, a scalar
    """

    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent))
    return loss



def my_loss_norm(output, target, latent, V, lam_sparse, lam_orthog, sample_weight):

    """
    Loss function when using orthogonality penalty instead of constraint

    Parameters
    ----------
    output: the predictions
        torch 2d tensor of size [n_time, output_size]
    target: ground truth output
        torch 2d tensor of size [n_time, output_size]
    latent: low dimensional representations
        torch 2d tensor of size [n_time, hidden_size]
    lam_sparse: sparsity penalty weight
        scalar
    lam_orthog: orthogonality regularization weight
        scalar
    sample_weight: weighting of each sample
        torch 2d tensor of size [n_time, 1]


    Returns
    -------
    loss: the value of the cost function, a scalar
    """

    loss = torch.sum((sample_weight*(output - target))**2) + lam_sparse*torch.sum(torch.abs(latent)) + lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1]))**2
    return loss



def my_loss_norm_poiss(pred_frs, spikes, latent, V, lam_sparse, lam_orthog, sample_weight, loss_sigmas, lam_loss):

    """
    Loss function when using orthogonality penalty instead of constraint

    Parameters
    ----------
    output: the predictions
        torch 2d tensor of size [n_time, output_size]
    target: ground truth output
        torch 2d tensor of size [n_time, output_size]
    latent: low dimensional representations
        torch 2d tensor of size [n_time, hidden_size]
    lam_sparse: sparsity penalty weight
        scalar
    lam_orthog: orthogonality regularization weight
        scalar
    sample_weight: weighting of each sample
        torch 2d tensor of size [n_time, 1]

    
    Returns
    -------
    loss: the value of the cost function, a scalar
    """
    term1 = -torch.sum(-gammaln(spikes+1)-pred_frs+spikes*torch.log(pred_frs+1e-8))
    term2 = lam_sparse*torch.sum(torch.abs(latent))
    term3 = lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1]))**2
    loss_sigmas2 = 1+torch.nn.functional.softplus(loss_sigmas)
    # term4 = lam_loss*torch.sum(0.5 * (latent[1:]-latent[:-1])**2 / (loss_sigmas**2) + torch.log(loss_sigmas)) # sum over one axis when diff sigma for each latent
    term4 = lam_loss*torch.sum(0.5 * (latent[1:]-latent[:-1])**2 / (loss_sigmas2**2) + torch.log(loss_sigmas2)) # sum over one axis when diff sigma for each latent
    # print("term1: ", term1, "term2: ", term2, "term3: ", term3, "term4: ", term4)

    return term1 + term2 + term3 # + term4