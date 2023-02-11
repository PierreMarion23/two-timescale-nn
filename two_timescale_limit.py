import numpy as np


def fast_solution(x, learner, target):
    """Returns the function that solves the perfect fast problem.
    
    The returned function is the best piecewise interpolant of target with subdivisions given by learner.
    """
    if x < learner.u[0]:
        return 0
    elif x >= learner.u[-1]:
        return np.sum(target.a)
    left_neuron = np.where(learner.u <= x)[0][-1]    # the closest neuron to the left of x
    right_neuron = np.where(learner.u > x)[0][0]     # the closest neuron to the right of x
    next_jumps = np.where(target.u >= learner.u[left_neuron])[0]
    if len(next_jumps) == 0:
        # the closest learner neuron to the left of x is after the last jump of the target
        return np.sum(target.a)
    else:
        next_jump = next_jumps[0]
    if target.u[next_jump] > learner.u[right_neuron]:
        #  there is no jump between left_neuron and right_neuron, thus solution(x) is equal to target(x).
        return np.sum(target.a[:next_jump])
    else:
        # x is next to a jump.
        distance_left = target.u[next_jump] - learner.u[left_neuron]
        distance_right = learner.u[right_neuron] - target.u[next_jump]
        if next_jump == 0:
            # special case: x is between zero and the first neuron of the learner
            # hence fast_solution(x) should be equal to the bias that is target.a[0].
            distance_left = 0
        return (np.sum(target.a[:next_jump]) * distance_left + np.sum(target.a[:next_jump+1]) * distance_right) / (distance_left + distance_right)


def slow_derivatives(distance_left, distance_right, neuron_weight):
    """Auxiliary function to implement the computation of derivatives wrt to the neurons' positions"""
    return (- (distance_right * neuron_weight / (distance_right + distance_left))**2, 
            (distance_left * neuron_weight / (distance_right + distance_left))**2)


def two_timescale_limit_derivative(learner, target):
    """Computes the derivatives wrt to the neurons' positions in the two timescale limit."""
    derivatives = np.zeros_like(learner.u)
    distance_to_jump = np.zeros_like(learner.u)
    # remove the dummy neuron target.u[0] that corresponds to the bias
    extended_target_subdivision = np.array([0] + list(target.u[1:]) + [1])  
    for k in range(len(extended_target_subdivision) - 1):
        # check if there are two neurons in each subdivision of the target
        neurons_in_subdivision = np.all([learner.u >= extended_target_subdivision[k], 
                                         learner.u <= extended_target_subdivision[k+1]], 
                                        axis=0)
        if np.sum(neurons_in_subdivision) < 2:
            raise ValueError('learner has less than two neurons in a target subdivision')

    for k in range(len(target.u) - 1):
        # The dummy neuron target.u[0] should not move, hence we start with target.u[1].
        jump = target.u[k+1]
        left_flanking_neuron = np.where(learner.u <= jump)[0][-1]
        right_flanking_neuron = np.where(learner.u >= jump)[0][0]
        distance_left = jump - learner.u[left_flanking_neuron]
        distance_right = learner.u[right_flanking_neuron] - jump
        if distance_left > 0 and distance_right > 0:
            derivative_left, derivative_right = slow_derivatives(distance_left, distance_right, target.a[k+1])
            derivatives[left_flanking_neuron] = derivative_left
            derivatives[right_flanking_neuron] = derivative_right
            distance_to_jump[left_flanking_neuron] = -distance_left
            distance_to_jump[right_flanking_neuron] = distance_right
    return derivatives, distance_to_jump
