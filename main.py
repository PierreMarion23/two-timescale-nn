import configs
import experiment

seed = 42

if __name__ == '__main__':
    # To reproduce the result of the paper, uncomment the next two lines
    experiment.run(configs.same_lr_config_paper, seed)             # approx. 2 minutes
    # experiment.run(configs.tt_config_paper, seed)        # approx. 4 hours
    # This experiment is similar to the previous one but runs quicker (albeit with more noise).
    # experiment.run(configs.tt_config_quick, seed)  # approx. 4.5 minutes
    # experiment.run_repeats(configs.tt_config_quick_repeats, seed)   # approx. 90 minutes
    # experiment.run_epsilon(configs.varying_epsilon, seed)  # approx. 10 hours
    pass
