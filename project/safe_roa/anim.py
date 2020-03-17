''' visualization and animations for 1d example

Ethan Lew
(elew@pdx.edu)
'''

import matplotlib.pyplot as plt
import numpy as np

from gproa import compute_v_dot_distribution
from example import lyapunov_function

import imageio

from glob import glob
import re

def _sort_first_digit(x):
    return int(re.search(r'\d+', x).group())

def get_files(path, sstring="*_1d_examples.png"):
    l = glob(path+"/"+ sstring)
    l.sort(key=_sort_first_digit)
    return l

def fig_to_gif(path, **kwargs):
    filenames = get_files(path, **kwargs)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(path+"/"+'1d_examples.gif', images)

def fig_model(path, idx, *args, **kwargs):
    # Create figure axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax_model(axes, *args, **kwargs)
    plt.savefig( path + "/" + f"{idx}_1d_examples.png")


def ax_model(axes, S, DynSys, beta = 2, legend=False):
    # Format axes
    axes[0].set_title('GP model of the dynamics')
    axes[1].set_xlim(DynSys.extent)
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel(r'Upper bound of $\dot{V}(x)$')
    axes[1].set_title(r'Determining stability with $\dot{V}(x)$')

    # Lyapunov function
    V, dV = lyapunov_function(DynSys.grid)
    mean, var = DynSys.gp._raw_predict(DynSys.grid)
    V_dot_mean, V_dot_var = compute_v_dot_distribution(dV, mean, var)

    # Plot dynamics
    axes[0].plot(DynSys.grid, DynSys.true_dynamics(DynSys.grid, noise=False), color='black', alpha=0.8)

    axes[0].fill_between(DynSys.grid[:, 0],
                         mean[:, 0] + beta * np.sqrt(var[:, 0]),
                         mean[:, 0] - beta * np.sqrt(var[:, 0]),
                         color=(0.8, 0.8, 1))
    axes[0].plot(DynSys.gp.X, DynSys.gp.Y, 'x', ms=8, mew=2)

    # Plot V_dot
    v_dot_est_plot = plt.fill_between(DynSys.grid.squeeze(),
                                      V_dot_mean + beta * np.sqrt(V_dot_var),
                                      V_dot_mean - beta * np.sqrt(V_dot_var),
                                      color=(0.8, 0.8, 1))
    threshold = plt.plot(DynSys.extent, [-DynSys.L * DynSys.tau, -DynSys.L * DynSys.tau], 'k-.', label=r'Safety threshold ($L \tau$ )')
    v_dot_true_plot = axes[1].plot(DynSys.grid, dV * DynSys.true_dynamics(DynSys.grid, noise=False), 'k',
                                   label=r'True $\dot{V}(x)$')

    # Create twin axis
    ax2 = axes[1].twinx()
    ax2.set_ylabel(r'$V(x)$')
    ax2.set_xlim(DynSys.extent)

    # Plot Lyapunov function
    V_unsafe = np.ma.masked_where(S, V)
    V_safe = np.ma.masked_where(~S, V)
    unsafe_plot = ax2.plot(DynSys.grid, V_unsafe, 'b', label=r'$V(x)$ (unsafe, $\dot{V}(x) > L \tau$)')
    safe_plot = ax2.plot(DynSys.grid, V_safe, 'r', label=r'$V(x)$ (safe, $\dot{V}(x) \leq L \tau$)')

    if legend:
        lns = unsafe_plot + safe_plot + threshold + v_dot_true_plot
        labels = [x.get_label() for x in lns]
        plt.legend(lns, labels, loc=4, fancybox=True, framealpha=0.75)

    # Create helper lines
    if np.any(S):
        x_safe = DynSys.grid[S][np.argmax(V[S])]
        y_range = axes[1].get_ylim()
        axes[1].plot([x_safe, x_safe], y_range, 'k-.')
        axes[1].plot([-x_safe, -x_safe], y_range, 'k-.')

