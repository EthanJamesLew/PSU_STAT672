''' Visualization and Animation of Pendulums

Ethan Lew
(elew@pdx.edu)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess
from matplotlib import cm
from matplotlib import patches

from anim import fig_to_gif

from gproa import *


def animate_pendulum(ax : plt.Axes, t : np.array, sol : np.array, l=1):
    ''' animate pendulum model
    :param t: (N) time value
    :param sol: (N, 2) odeint solution
        sol[:, 0] is the angular position, sol[:, 1] is the angular rate
    :return: list of plot elements
    '''

    plt.rcParams['font.size'] = 15
    lns = []
    for i in range(len(sol)):
        ln, = ax.plot([0, l*np.sin(sol[i, 0] + np.pi)], [0, -l*np.cos(sol[i, 0] + np.pi)],
                      color='k', lw=2)
        pt = ax.scatter([l*np.sin(sol[i, 0] + np.pi)], [-l*np.cos(sol[i, 0] + np.pi)], s=100, c='k')
        tm = ax.text(-l, 0.9*l, 'time = %.1fs' % t[i])
        lns.append([ln, pt, tm])
    margin = np.abs(l) * 0.05
    ax.set_xlim(-l - margin, l + margin)
    ax.set_ylim(-l -margin, l + margin)
    ax.grid()
    return lns

def animate_waveforms(ax : plt.Axes, t : np.array, sol : np.array):
    ''' animate waveforms over time
    :param ax: plt.axis
    :param t: (N) times
    :param sol: (N, 2) pendulum solution
    :return: list of line plots
    '''
    lns = []
    for i in range(len(sol)):
        ln = ax.plot(t[:i], sol[:i, 0], 'g')
        ln1 = ax.plot(t[:i], sol[:i, 1], 'b')
        lns.append([*ln, *ln1])
    ax.legend(['position', 'rate'])
    ax.set_xlim(0, t.max())
    ax.set_ylim(sol.min(), sol.max())
    ax.grid()
    return lns

def animate_multiplot(t : np.array, sol : np.array):
    ''' animate pendulum and plot together
    :param t: (N) times
    :param sol: (N, 2) pendulum state solution
    :return: figure, animation
    '''
    fig, ax =plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    pens = animate_pendulum(ax[0], t, sol)
    lns = animate_waveforms(ax[1], t, sol)
    ani = merge_ax(pens, lns)
    return fig, ani

def merge_ax(ax0 : list, ax1 : list):
    return [[*ax0[i], *ax1[i]] for i in range(0, len(ax0))]

def ani_to_file(fig : plt.Figure, lns : list, fn='odeint_single_pendulum_artistanimation'):
    ani = animation.ArtistAnimation(fig, lns, interval=50)
    ani.save('./img/'+fn+'.mp4',writer='ffmpeg', fps=1000/50)
    ani.save('./img/'+fn+'.gif',writer='imagemagick', fps=1000/50)

def plot_inv_pend():
    import pickle
    with open('inv_pend_data.p', 'rb') as fp:
        data = pickle.load(fp)
    Ss = data['Ss']
    V = data['V']
    states = data['states']
    accuracy = data['accuracy']
    Pn = data["Pn"]
    c_prior = data["c_prior"]
    c_true = data['c_true']
    Tx = data['Tx']
    print(states)
    states = np.array(states)
    states = np.squeeze(states, axis=1)
    print(states.shape)
    for i in range(100):
        X = states[:i+1, :]
        plot_S(i, Ss[i], X, V, accuracy, Pn, c_prior, c_true, Tx)


def plot_S(idx, S, X, V, accuracy, Pn, c_prior, c_true, Tx):
    def denorm_ellipse(P, level):
        """Return the ellipse _bounds, but denormalized."""
        x0, x1_u, x1_l = ellipse_bounds(P, level)
        return Tx[0, 0] * x0, Tx[1, 1] * x1_u, Tx[1, 1] * x1_l

    def denormalize_x(x):
        """Denormalize x vector"""
        x = np.asarray(x)
        return x.dot(Tx)
    fig, ax = plt.subplots()
    c_est = find_max_levelset(S, V, accuracy)
    colors = ['b', 'm', 'r']

    plt.fill_between(*denorm_ellipse(Pn, c_prior), color=colors[0], alpha=0.5)
    plt.fill_between(*denorm_ellipse(Pn, c_true), color=colors[1], alpha=0.5)
    plt.fill_between(*denorm_ellipse(Pn, c_est), color=colors[2], alpha=0.5)

    patch0 = patches.Patch(color=colors[0], alpha=0.5, label='Prior safe set')
    patch1 = patches.Patch(color=colors[1], alpha=0.5, label='True safe set')
    patch2 = patches.Patch(color=colors[2], alpha=0.5, label='Estimated safe set')

    legs = [patch0, patch1, patch2]
    labels = [x.get_label() for x in legs]
    leg = plt.legend(legs, labels, loc=3, borderaxespad=0)

    data = denormalize_x(X[1:, :])
    plt.plot(data[:, 0], data[:, 1], 'xk')

    plt.xlabel(r'Angle $\theta$')
    plt.ylabel(r'Angular velocity $\dot{\theta}$')
    plt.savefig('./img/tempp/' + f'{idx}_inv_pend.png')
    #plt.show()


def plot_paths():
    def denormalize_x(x):
        """Denormalize x vector"""
        x = np.asarray(x)
        return x.dot(Tx)
    import pickle
    with open('./pend_paths.p', 'rb') as fp:
       data = pickle.load(fp)
    t, sol1, sol2 = data['t'], data['sol1'], data['sol2']
    with open('inv_pend_data.p', 'rb') as fp:
       data = pickle.load(fp)
    Tx = data['Tx']
    ani_to_file(*animate_multiplot(t, sol1), fn="stable_path")


if __name__ == "__main__":
    plot_paths()
    #plot_inv_pend()
    #fig_to_gif('./img/tempp', sstring="*_inv_pend.png")
    #t = np.linspace(0, 10, 100)
    #sol = np.zeros((len(t), 2))
    #sol[:, 0] = np.sin(t)
    #sol[:, 1] = np.cos(t)
    #ani_to_file(*animate_multiplot(t, sol))
