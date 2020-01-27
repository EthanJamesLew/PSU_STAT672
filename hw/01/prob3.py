
import numpy as np
#import scipy.special.jv 

##KERNELS
def k_polynomial(x, xp, d):
    return (np.dot(x, xp)+1)**d


def k_gaussian(x, xp, sigma):
    return np.exp(-np.sum((x-xp)**2)/(2*(sigma**2)))


def k_tanh(x, xp, kappa, Theta):
    return np.tanh(kappa * np.dot(x, xp) + Theta)

#def k_bessel(x, xp, n, v, sigma):
#    return jv(v, sigma*np.sum((x-xp)**2))/(np.sum((x-xp)**2)**(n*(v+1)))

def kernel_mat(f, x):
    '''
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

def sample_functions(m, N, kernel):
    ''' For X={-m,...,1,0,1,...,m}, sample functions based on kernel 
    :param m: +/- X range
    :param N: number of samples 
    :param kernel: kernel function K(x,y)
    '''
    # Sample Functions
    X = np.arange(-m, m+1) 
    K = kernel_mat(kernel, X)
    return X, np.random.multivariate_normal(np.zeros((2*m+1)), K, (N))

def plot_samples(ax, f, title):
    # Plot Results
    for fi in f:
        ax.plot(X, fi)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f_i(x)")
    ax.set_xlim((np.min(X), np.max(X)))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    m = 10 # +/- range of X
    tau = 2 # tau Gaussian kernel parameter
    N = 10 # Number of samples
    
    X, f0 = sample_functions(m, N, lambda x,y: k_gaussian(x, y, 3))
    X, f1 = sample_functions(m, N, lambda x,y: k_gaussian(x, y, 10))
    X, f2 = sample_functions(m, N, lambda x,y: k_polynomial(x, y, 2))
    #X, f3 = sample_functions(m, N, lambda x,y: k_bessel(x, y, 2,1,1))
    
    # Plot Results
    figg, axg = plt.subplots((2))
    figk, axk = plt.subplots((2))
    plot_samples(axg[0], f0, "Samples for tau = 3")
    plot_samples(axg[1], f1, "Samples for tau = 10")
    
    plot_samples(axk[0], f2, "Samples for Polynomial Kernel d=2")
    plot_samples(axk[0], f2, "Samples for Polynomial Kernel d=2")

    plt.show()

