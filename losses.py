import torch
import torch.nn.functional as F

def gaussian_elbo(x1,x2,z,sigma,mu,logvar):
    
    #
    # Problem 5b: Compute the evidence lower bound for the Gaussian VAE.
    #             Use the closed-form expression for the KL divergence from Problem 1.
    #

    return reconstruction, divergence

def mc_gaussian_elbo(x1,x2,z,sigma,mu,logvar):

    #
    # Problem 5c: Compute the evidence lower bound for the Gaussian VAE.
    #             Use a (1-point) monte-carlo estimate of the KL divergence.
    #

    return reconstruction, divergence

def cross_entropy(x1,x2):
    return F.binary_cross_entropy_with_logits(x1, x2, reduction='sum')/x1.shape[0]

def discrete_output_elbo(x1,x2,z,logqzx):

    #
    # Problem 6b: Compute the evidence lower bound for a VAE with binary outputs.
    #             Use a (1-point) monte carlo estimate of the KL divergence.
    #

    return reconstruction, divergence
