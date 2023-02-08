import distutils.spawn
import os

from matplotlib import rc
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns


sns.set(font_scale=1.5)
if distutils.spawn.find_executable('latex'):
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

markers = ['', 'o', 'v', 's', '+', 'x']
loss_markevery = [0, 9, 11, 13, 15, 17]
plot_markevery = [57, 60, 64, 71]


def plot_current_iterate(iterate_number, xs, ys, labels, ylim, neuron_positions=[], legend='', 
                         show=False, save=False, folderpath='', file_prefix='plot'):
    """Plots several neural networks. Each neural network is specified by a list of (x, y) coordinates which
       describe its graph, and optionally by a list of neuron positions.

    :param xs: list of lists of x-coordinates
    :param ys: list of lists of y-coordinates (should be of the same shape than xs)
    :param labels: list of labels (should be of the same length than xs)
    :param yim: ylim of the plot
    :param neuron_positions: list of lists of neuron positions
    :param legend: TODO: describe this
    :param show: if True, shows the plot
    :param save: if True, saves the plot under 'folderpath/plot{iteration_number}{_leg}.png'
                 where {_leg} is present for plots with legents
    :param folderpath: folder in which to save the plot
    """
    if len(xs) != len(ys) or len(xs) != len(labels):
        raise ValueError('lists should have the same length.')
    if folderpath and not os.path.exists(folderpath):
        os.makedirs(folderpath)
    for k in range(len(xs)):
        plt.plot(xs[k], ys[k], label=labels[k], marker=markers[k], markevery=plot_markevery[k])
    for idx_neuron_batch in range(len(neuron_positions)):
        plt.vlines(neuron_positions[idx_neuron_batch],
                    ymin=ylim[0],
                    ymax=ylim[1],
                    color='C{}'.format(idx_neuron_batch+1),
                    linestyle='--',
                    alpha=0.4)
    plt.ylim(ylim)
    if legend['print']:
        plt.legend(**legend['parameters'])
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(folderpath, '{}_{}.png'.format(file_prefix, iterate_number)), dpi=150, bbox_inches='tight')
    plt.close()


def plot_losses(nb_steps, losses_lists, labels, ylim, show=False, save=False, folderpath='', file_prefix='loss'):
    """Plots several losses across iterations.

    :param nb_steps: number of iterations
    :param losses_lists: list of losses. Each element of the list must be an iterable of size nb_steps.
    :param labels: labels for each loss
    :param yim: ylim of the plot
    :param show: if True, shows the plot
    :param save: if True, saves the plot under 'folderpath/loss.png'
    :param folderpath: folder in which to save the plot
    """
    if len(losses_lists) != len(labels):
        raise ValueError('lists should have the same length.')
    if len(losses_lists) > 5:
        raise ValueError('can plot at most 5 losses at once.')
    for idx in range(len(losses_lists)):
        plt.plot(np.linspace(0, nb_steps, num=len(losses_lists[idx])),
                 losses_lists[idx],
                 label=labels[idx],
                 color='C{}'.format(idx+1),
                 marker=markers[idx+1],
                 markevery=loss_markevery[idx+1])
    plt.yscale('log')
    if len(losses_lists) == 1:
        plt.xlabel(r'$p$')
    elif len(losses_lists) == 2:
        plt.xlabel(r'$p = \tau / \varepsilon h$')
    plt.ylabel(r'$L$')
    plt.legend(loc='upper right')
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(folderpath, '{}.png'.format(file_prefix)), dpi=150, bbox_inches='tight')
    plt.close()

def render_gif(folderpath, file_prefix, nb_figures):
    """Renders GIF from figures.
    
    Figures should be under 'folderpath/plot{idx}_leg.png' for idx from 0 to nb_figures-1.
    Saves the GIF under 'folderpath/plot_leg.gif'.
    """
    images = []
    for k in range(nb_figures):
        im = Image.open(os.path.join(folderpath, '{}_{}.png').format(file_prefix, k))
        images.append(im)
    images[0].save(os.path.join(folderpath, '{}.gif').format(file_prefix), save_all=True, append_images=images[1:], optimize=False, duration=50, loop=0)
