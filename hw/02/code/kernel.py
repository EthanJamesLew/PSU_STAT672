
from scipy.special import jv 
import numpy as np

##KERNELS
def k_polynomial(x, xp, d):
    return (np.dot(x, xp)+1)**d


def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x-xp)**2)/(2*(sigma**2)))


def k_tanh(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

def k_bessel(x, xp, n, v, sigma):
    return jv(v, sigma*np.sum((x-xp)**2))/(np.sum((x-xp)**2)**(-float(n*(v+1))))

def k_min(x, xp):
    return np.min([x, xp])

def kernel_mat(f, x):
    ''' Given a kernel f and a collection of observations x, construct the matrix
    [K]_{ij} = f(x_i, x_j)
    :param f: kernel function
    :param x: vector of values
    :return: K symmetric matrix
    '''
    n = len(x)
    K = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i+1):
            v = f(x[i], x[j])
            K[i, j] = v
            K[j, i] = v
    return K
