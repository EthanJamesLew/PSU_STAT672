''' Kernel PCA Implementation 

Ethan Lew
(elew@pdx.edu)
'''
import numpy as np 
from inspect import signature 

from kernel import * 

class KernelPCA:
    def __init__(self):
        # observations
        self.X = None

        # kernel
        self._kernel = lambda x,y: np.dot(x,y) 

        # dimension of observations
        self.ndim = None 

        # store the principal component transform
        self.U = None
        self.L = None 

    def add_observations(self, X):
        ''' Add n observations of dimension d for training
        :param X (n x d)
        '''
        if self.X is None:
            self.ndim = X.shape[1]
            self.X = X
        else:
            if self.X.shape[1] != self.ndim:
                raise Exception("Observations must be of {0} dimension!".format(self.ndim))
            self.X = np.vstack((self.X, X))

    @property
    def observations(self):
        return self.X

    @observations.setter
    def obervations(self, X):
        self.add_observations(X)

    def _check_kernel(self, kernel):
        ''' Make sure kernel is of two arguments
        '''
        if len(signature(kernel).parameters) != 2:
            raise Exception("kernel must have two arguments!")

    def add_kernel(self, kernel):
        self._check_kernel(kernel)
        self._kernel = kernel 

    @property
    def kernel(self):
        return self._kernel  

    @kernel.setter
    def kernel(self, kernel):
        self.add_kernel(kernel)

    def train(self):
        ''' Perform K-PCA
        Calculate centering kernel, perform eigendecomposition
        and store the results
        '''
        if self.X is None:
            raise Exception("No training data is available for KPCA")
        # construct kernel matrix
        K = kernel_mat(self.kernel, self.X)
        
        # make a kernel from centered data
        n = self.X.shape[0]
        I = np.eye(n)
        D = np.ones((n,n))
        Kc = (I-D) @ K @ (I-D)
        
        # perform eigendecomposition
        L,U = np.linalg.eig(Kc) 

        # store the results
        self.U = U
        self.L = L 

    def transform(self, Xp):
        if self.U is None:
            raise Exception("KPCA hasn't been trained yet!")
        if Xp.shape[1] != self.ndim:
            raise Exception("X value dimension don't match training data dimension!")
        return self.U.T @ Xp

    def get_U(self):
        return self.U

    def __call__(self, Xp):
        return self.transform(Xp)
    



