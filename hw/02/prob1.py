
import numpy as np
from scipy.special import jv 
import matplotlib.pyplot as plt

# For LaTeX Plots
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

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

def sample_functions(X, N, K):
    ''' For X={-m,...,1,0,1,...,m}, sample functions based on kernel 
    :param X: observations
    :param N: number of samples 
    :param kernel: kernel matrix satisfying [K]_{ij} = K(x_i, x_j)
    '''
    # Sample Functions
    return X, np.random.multivariate_normal(np.zeros(X.shape), K, (N))

def plot_samples(ax, X, f, title):
    """ Given a plt.axis object, plot f and give it a title 
    :param ax: pyplot axis object
    :param f: n samples of length l
    :param title: string for axis title 
    """
    # Plot Results
    for fi in f:
        ax.plot(X, fi)
    ax.set_title(title)
    ax.set_ylabel(r"$f_i(x)$")
    ax.set_xlim((np.min(X), np.max(X)))

if __name__ == "__main__":
    m = 10 # +/- range of X
    tau = 10 # tau Gaussian kernel parameter
    N = 10 # number of samples to draw
    
    n = m + 1
    X = np.arange(-m, 1) 
    K = kernel_mat(lambda x,y: k_gaussian(x, y, tau), X)
    D = np.ones((n,n)) / n
    I = np.eye(n)
    Kc = (I - D) @ K @ (I - D)
    X, f0 = sample_functions(X, N, Kc)
    
    # check that the samples are centered
    assert(np.any(np.finfo(np.float32).eps > f0.sum(axis=1)/n))
    
    # plot results
    figg, axg = plt.subplots()
    plot_samples(axg, X, f0, r"Samples for $\tau = 10$")
    axg.set_xlabel(r"$x$")

    plt.show()
