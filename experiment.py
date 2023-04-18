import copy
import os
import pickle
import time

import numpy as np

import two_timescale_limit
from neural_network import NeuralNetwork
import plots


def run(config, seed=None):
    """Main run function."""
    start_time = time.time()
    print('Beginning experiment {}'.format(config['name']))

    folderpath = config['folderpath']
    if folderpath and not os.path.exists(folderpath):
        os.makedirs(folderpath)
    target_weights = config['target_weights']
    target_positions = config['target_positions']
    nb_neurons = config['nb_neurons']
    tmax = config['tmax']         # integration time such that nb_steps = int(tmax / (h * epsilon))
    h = config['h']
    epsilon = config['epsilon']
    eta = config['eta']
    noise_magnitude = config['noise_magnitude']
    run_tt_limit = config['run_tt_limit']
    plot_iterates = config['plot_iterates']
    plot_losses = config['plot_losses_one_run']

    if seed:
        np.random.seed(seed)
    nb_loss_eval = 100              # number of steps when the loss is evaluated and the functions are plotted
    nb_limit_integration = 1000     # number of steps when the limit network is updated
    X = np.linspace(0, 1, 1000)     # for plots

    # The piecewise constant target is implemented as a NN with eta << 1.
    target = NeuralNetwork(target_weights, target_positions, eta=1e-8)

    # Initial subdivision for learners.
    subdivision = np.sort(np.random.rand(nb_neurons))
    sgd_learner = NeuralNetwork(np.zeros(nb_neurons+1), subdivision, eta)
    limit_learner = NeuralNetwork(np.zeros(nb_neurons+1), subdivision, eta)

    nb_steps = int(tmax / (h * epsilon))
    print('Number of steps: {}'.format(nb_steps))
    interval_loss_eval = int(nb_steps / nb_loss_eval)
    interval_limit_integration = int(nb_steps / nb_limit_integration)
    limit_integration_stepsize = tmax / nb_limit_integration    # Euler step size for limit dynamics
    sgd_losses = []
    limit_losses = []
    legends = [{'name': 'no_legend', 'print': False},
               {'name': 'outside_legend', 'print': True, 'parameters': {'loc': 'lower left', 'bbox_to_anchor': (-0.6, 0.27)}},
               {'name': 'inside_legend', 'print': True, 'parameters': {'loc': 'upper right'}}]

    for step in range(nb_steps+1):
        if step % interval_loss_eval == 0:
            sgd_losses.append(target.loss(sgd_learner))
            limit_losses.append(target.loss(lambda x: two_timescale_limit.fast_solution(x, limit_learner, target)))

            if plot_iterates:
                xs = [X, X]
                ys = [[target(x) for x in X], [sgd_learner(x) for x in X]]
                labels = ['Target', 'SGD']
                neuron_positions = [sgd_learner.u]
                for legend in legends:
                    plots.plot_current_iterate(iterate_number=int(step / interval_loss_eval),
                        xs=xs, 
                        ys=ys, 
                        labels=labels,
                        ylim=[min(0, min(ys[0]))-0.2, max(0, max(ys[0]))+0.2],  # this includes y=0 in the plot.
                        neuron_positions=neuron_positions,
                        legend=legend,
                        save=True,
                        folderpath=folderpath,
                        file_prefix='plot_sgd_only_{}'.format(legend['name']))
                if run_tt_limit:
                    xs.append(X)
                    ys.append([two_timescale_limit.fast_solution(x, limit_learner, target) for x in X])
                    labels.append('TT limit')
                    for legend in legends:
                        plots.plot_current_iterate(iterate_number=int(step / interval_loss_eval),
                                xs=xs, 
                                ys=ys, 
                                labels=labels,
                                ylim=[min(0, min(ys[0]))-0.2, max(0, max(ys[0]))+0.2], # this includes y=0 in the plot.
                                legend=legend,
                                save=True,
                                folderpath=folderpath,
                                file_prefix='plot_with_limit_{}'.format(legend['name']))
            print('Percentage done: {0:.1f}%'.format(100 * step / nb_steps))
        
        x = np.random.rand()
        noise = noise_magnitude * (2 * np.random.rand() - 1)
        grads_a = sgd_learner.stoch_grad_a_loss(target, x, noise)
        grads_u = sgd_learner.stoch_grad_u_loss(target, x, noise)
        sgd_learner.a -= h * grads_a
        sgd_learner.u -= h * epsilon * grads_u

        if run_tt_limit and step % interval_limit_integration == 0:
            try:
                derivatives, distance_to_kink = two_timescale_limit.two_timescale_limit_derivative(limit_learner, target)
            except two_timescale_limit.BadSubdivision:
                print('Learner has less than two neurons in a target subdivision at step {}'.format(step))
                raise two_timescale_limit.BadSubdivision
            for i in range(len(limit_learner.u)):
                if np.abs(distance_to_kink[i]) < np.abs(limit_integration_stepsize * derivatives[i]):
                    # if the step is greater than the distance to the kink: jump to the kink.
                    # this is a decent (first order) approximation to discontinuous dynamics.
                    limit_learner.u[i] -= distance_to_kink[i]
                else:
                    limit_learner.u[i] -= limit_integration_stepsize * derivatives[i]
    
    if plot_iterates:
        print('Rendering GIF...')
        plots.render_gif(folderpath, 'plot_sgd_only_outside_legend', int(nb_steps / interval_loss_eval))
        if run_tt_limit:
            plots.render_gif(folderpath, 'plot_with_limit_outside_legend', int(nb_steps / interval_loss_eval))
        
    # Plot loss as a function on the number of steps.
    losses = [sgd_losses] if not run_tt_limit else [sgd_losses, limit_losses]

    if plot_losses:
        labels = ['SGD'] if not run_tt_limit else ['SGD', 'Two-timescale limit']
        plots.plot_losses(nb_steps,
            np.array([losses]),
            labels,
            show=False,
            save=True,
            folderpath=folderpath)

    print('Experiment time: {0:.0f} seconds'.format(time.time()-start_time))
    print()

    return nb_steps, losses


def run_repeats(config):
    """Repeats a certain number of runs."""
    n_repeats = config['n_repeats']

    print('Repeating experiment {} times'.format(n_repeats))
    start_time = time.time()

    all_losses = []
    k = 0
    while k < n_repeats:
        print('Beginning repeat number {}'.format(k))
        try:
            _, losses = run(config)
            all_losses.append(losses)
            k += 1
        except two_timescale_limit.BadSubdivision:
            print('Error during the run. Re-running with another random seed.')

    print('Experiment time for all repeats: {0:.0f} seconds'.format(time.time()-start_time))
    print()

    return all_losses


def run_epsilon(config, seed):
    """Repeats a certain number of runs for varying values of epsilon."""
    folderpath = config['folderpath']
    if folderpath and not os.path.exists(folderpath):
        os.makedirs(folderpath)
    epsilons = config['epsilon']

    start_time = time.time()

    np.random.seed(seed)

    all_losses_per_epsilon = []
    for epsilon in epsilons:
        print('Starting experiments for epsilon = {0:.2f}'.format(epsilon))
        local_config = copy.copy(config)
        local_config['epsilon'] = epsilon
        all_losses = run_repeats(local_config, save=False)
        all_losses_per_epsilon.append(all_losses)

    with open(os.path.join(folderpath, 'losses.pickle'), 'wb') as handle:
        pickle.dump(all_losses_per_epsilon, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plots.plot_boxplot(epsilons, 
                       np.array(all_losses_per_epsilon)[:,:,0,-1],  # select SGD loss at last time step
                       show=False, 
                       save=True, 
                       folderpath=folderpath)

    print('Total experiment time: {0:.0f} seconds'.format(time.time()-start_time))
    print()
