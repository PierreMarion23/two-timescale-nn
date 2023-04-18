import numpy as np


same_lr_config_paper = {
    'name': 'same-lr-exp',
    'folderpath': 'figures/same_lr',
    'target_weights': np.array([1, 4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 10,
    'h': 1e-5,
    'epsilon' : 1,
    'eta': 4e-3,
    'noise_magnitude': 1,
    'run_tt_limit': False,
    'plot_iterates': True,
    'plot_losses_one_run': True,
}


tt_config_paper = {
    'name': 'tt-exp',
    'folderpath': 'figures/tt',
    'target_weights': np.array([1, 4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 0.035,
    'h': 1e-5,
    'epsilon' : 2e-5,
    'eta': 4e-3,
    'noise_magnitude': 1,
    'run_tt_limit': True,
    'plot_iterates': True,
    'plot_losses_one_run': True,
}


tt_config_quick = {
    'name': 'tt-exp-quick',
    'folderpath': 'figures/tt-quick',
    'target_weights': np.array([1, 4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 0.035,
    'h': 1e-3,
    'epsilon' : 2e-5,
    'eta': 4e-3,
    'noise_magnitude': 1,
    'run_tt_limit': True,
    'plot_iterates': True,
    'plot_losses_one_run': True,
}


varying_epsilon = {
    'name': 'varying-epsilon',
    'folderpath': 'figures/varying-epsilon',
    'target_weights': np.array([1, 4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 3,
    'h': 1e-5,
    'epsilon' : [0.01, 0.1, 1.0],
    'eta': 4e-3,
    'noise_magnitude': 1,
    'run_tt_limit': False,
    'plot_iterates': False,
    'plot_losses_one_run': False,
    'n_repeats': 20
}
