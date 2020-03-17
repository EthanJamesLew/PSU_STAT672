''' 1D Example Methods

Ethan Lew
(elew@pdx.edu)

'''

import GPy
import numpy as np
from gproa import sample_gp_function, quadratic_lyapunov_function
from gproa import compute_v_dot_distribution, compute_v_dot_upper_bound, \
    get_safe_set, find_max_levelset


class DynamicalSystem:
    def __init__(self):
        # Discretization constant
        tau = 0.001

        # x_min, x_max, discretization
        grid_param = [-1., 1., tau]

        extent = np.array(grid_param[:2])

        # Create a grid
        grid = np.arange(*grid_param)[:, None]
        num_samples = len(grid)

        # Observation noise
        noise_var = 0.01 ** 2

        # Mean dynamics
        mf = GPy.core.Mapping(1, 1)
        mf.f = lambda x: -0.25 * x
        mf.update_gradients = lambda a, b: None

        # Define one sample as the true dynamics
        np.random.seed(5)

        # Define the kernel
        kernel = GPy.kern.Matern32(1, lengthscale=0.2, variance=0.2**2) * GPy.kern.Linear(1)

        true_dynamics = sample_gp_function(kernel,
                                           [extent],
                                           num_samples=100,
                                           noise_var=noise_var,
                                           interpolation='kernel',
                                           mean_function=mf.f)

        gp = GPy.core.GP(np.array([[0]]), np.array([[0]]),
                         kernel, GPy.likelihoods.Gaussian(variance=noise_var),
                         mean_function=mf)

        V, dV = lyapunov_function(grid)

        # V, dV = lyapunov(grid)
        accuracy = np.max(V) / 1e10
        beta = 2

        # Lipschitz constant
        L_dyn = 0.25 + beta * np.sqrt(gp.kern.Mat32.variance) / gp.kern.Mat32.lengthscale * np.max(np.abs(extent))
        B_dyn = (0.25 + np.sqrt(gp.kern.Mat32.variance)) * np.max(np.abs(extent))
        B_dV = L_V = np.max(dV)
        L_dV = 1

        L = B_dyn * L_dV + B_dV * L_dyn

        self._extent = extent
        self._grid = grid
        self._true_dynamics = true_dynamics
        self._tau = tau
        self._L = L
        self._gp = gp
        self._dV = dV
        self._V = V
        self._beta = beta
        self._accuracy = accuracy

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def dV(self):
        return self._dV

    @property
    def V(self):
        return self._V

    @property
    def gp(self):
        return self._gp

    @property
    def extent(self):
        return self._extent

    @property
    def grid(self):
        return self._grid

    @property
    def true_dynamics(self):
        return self._true_dynamics

    @property
    def tau(self):
        return self._tau

    @property
    def L(self):
        return self._L

    @property
    def beta(self):
        return self._beta

def lyapunov_function(x):
    return quadratic_lyapunov_function(x, np.array([[1]]))

def get_max_level_set(dynamics_mean, dynamics_var, DynSys, S0):
    V_dot = compute_v_dot_upper_bound(DynSys.dV, dynamics_mean, dynamics_var, beta=2.)
    S = get_safe_set(V_dot, -DynSys.L*DynSys.tau, S0=S0)
    c = find_max_levelset(S, DynSys.V, DynSys.accuracy)
    S[:] = DynSys.V <= c
    return S

def update_gp(DynSys, S0):
    dynamics_mean, dynamics_var = DynSys.gp._raw_predict(DynSys.grid)
    S = get_max_level_set(dynamics_mean, dynamics_var, DynSys, S0)
    max_id = np.argmax(dynamics_var[S])
    max_state = DynSys.grid[S][[max_id], :].copy()
    DynSys.gp.set_XY(np.vstack([DynSys.gp.X, max_state]),
              np.vstack([DynSys.gp.Y, DynSys.true_dynamics(max_state, noise=True)[:, [0]]]))
    return S

def generate_images(N=100, path='.'):
    d = DynamicalSystem()
    mean, var = d.gp._raw_predict(d.grid)
    V_dot_mean, V_dot_var = compute_v_dot_distribution(d.dV, mean, var)
    S = get_safe_set(V_dot_mean + d.beta * np.sqrt(V_dot_var),
                     -d.L * d.tau,
                     S0=None)

    S0 = np.abs(d.grid.squeeze()) < 0.2

    fig_model(path, 0, S0, d)
    # Update the GP model a couple of times
    for idx in range(N):
        update_gp(d, S0)
        S = get_max_level_set(*d.gp._raw_predict(d.grid), d, S0)
        fig_model(path, idx + 1, S, d)
    fig_to_gif(path)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from anim import *

    generate_images(N=100, path='./img/temp')

