import torch
from torch.special import gammaln

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



def my_loss_norm_poiss(pred_frs, spikes, latent, V, lam_sparse, lam_orthog, sample_weight):

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
    '''
    poisson only defined over !positive!(or non negative) numbers- probability that we'll generate a certain number of spikes
    "soft plus" to end during the forwarding of model possibly 
    - clamping data? maybe not preferred but a possibility
    - centered data over 0 as mean- causing issues with negative values
    '''
    term2 = lam_sparse*torch.sum(torch.abs(latent))
    term3 = lam_orthog*torch.norm(V.T@V-torch.eye(V.shape[1]))**2
    # print("v shape: ", V.shape) #[50,8]
    # print(term1, term2, term3)
    return term1 + term2 + term3