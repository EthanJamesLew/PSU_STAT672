''' Visualization and Animation of Pendulums

Ethan Lew
(elew@pdx.edu)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess


def animate_pendulum(ax, t, sol, l=1):
    ''' animate pendulum model
    :param t: (N) time value
    :param sol: (N, 2) odeint solution
        sol[:, 0] is the angular position, sol[:, 1] is the angular rate
    :return: list of plot elements
    '''

    plt.rcParams['font.size'] = 15
    lns = []
    for i in range(len(sol)):
        ln, = ax.plot([0, l*np.sin(sol[i, 0])], [0, -l*np.cos(sol[i, 0])],
                      color='k', lw=2)
        pt = ax.scatter([l*np.sin(sol[i, 0])], [-l*np.cos(sol[i, 0])], s=100, c='k')
        tm = ax.text(-l, 0.9*l, 'time = %.1fs' % t[i])
        lns.append([ln, pt, tm])
    margin = np.abs(l) * 0.05
    ax.set_xlim(-l - margin, l + margin)
    ax.set_ylim(-l -margin, l + margin)
    ax.grid()
    return lns

def animate_waveforms(ax, t, sol):
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

def merge_ax(ax0, ax1):
    return [[*ax0[i], *ax1[i]] for i in range(0, len(ax0))]

def ani_to_file(fig, lns, fn='odeint_single_pendulum_artistanimation'):
    ani = animation.ArtistAnimation(fig, lns, interval=50)
    ani.save(fn+'.mp4',writer='ffmpeg', fps=1000/50)
    ani.save(fn+'.gif',writer='imagemagick', fps=1000/50)

if __name__ == "__main__":
    #fig = plt.figure(figsize=(5, 5), facecolor='w')
    #ax = fig.add_subplot(1, 1, 1)
    fig, ax =plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    t = np.linspace(0, 10, 100)
    sol = np.zeros((len(t), 2))
    sol[:, 0] = np.sin(t)
    sol[:, 1] = np.cos(t)
    pens = animate_pendulum(ax[0], t, sol)
    lns = animate_waveforms(ax[1], t, sol)
    ani = merge_ax(pens, lns)
    ani_to_file(fig, ani)
