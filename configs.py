import numpy as np


same_lr_config_paper = {
    'name': 'same-lr-exp',
    'folderpath': 'figures/same-h-final/same_lr',
    'target_weights': np.array([4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 10,
    'h': 1e-5,
    'epsilon' : 1,
    'eta': 4e-3,
    'run_tt_limit': False,
    'plot_iterates': True
}


tt_config_paper = {
    'name': 'tt-exp',
    'folderpath': 'figures/same-h-final/tt',
    'target_weights': np.array([4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 0.035,
    'h': 1e-5,
    'epsilon' : 2e-5,
    'eta': 4e-3,
    'run_tt_limit': True,
    'plot_iterates': True
}


tt_config_quick = {
    'name': 'tt-exp-quick',
    'folderpath': 'figures/same-h-final/tt-quick',
    'target_weights': np.array([4, -4, 2, -2, 4]),
    'target_positions': np.linspace(0.2, 0.8, 5, endpoint=True),
    'nb_neurons': 20,
    'tmax': 0.035,
    'h': 1e-3,
    'epsilon' : 2e-5,
    'eta': 4e-3,
    'run_tt_limit': True,
    'plot_iterates': True
}
