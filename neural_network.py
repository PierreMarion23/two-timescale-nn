import numpy as np
import scipy.integrate as integ


def sigma(x):
    return np.minimum(np.maximum((4*(x+0.5)**3), 0), 0.5) + np.minimum(np.maximum((0.5 - 4*(0.5-x)**3), 0), 0.5)


class NeuralNetwork:

    def __init__(self, a, u, eta):
        if len(a) != len(u) + 1:
            raise ValueError('there should be one more weight than position to account for the bias.')
        self.a = np.copy(a)
        self.u = np.zeros(len(self.a))
        self.u[0] = - eta / 2   # the bias is implemented as a neuron at position -eta/2.
        self.u[1:] = np.copy(u)
        if list(self.u) != list(np.sort(self.u)):
            raise ValueError('the neuron positions should be sorted.')
        self.eta = eta
        self.sigma = lambda x: sigma(x / self.eta)
        self.X = np.linspace(0, 1, 1000)   # for plots


    def __call__(self, x):
        """Evaluates the neural network at point x."""
        return np.dot(self.sigma(x - self.u), self.a)
        

    def loss(self, func):
        """Computes the L2 loss between self and an arbitrary function.
        
        The integration is divided along self.u, in order to limit integration warnings.
        """
        result = 0
        subdiv = [0] + list(self.u) + [1]
        for k in range(len(subdiv) - 1):
            result += integ.quad(lambda x: (self(x) - func(x))**2, subdiv[k], subdiv[k+1], limit=200)[0] 
        return result
    

    def stoch_grad_a_loss(self, teacher, x=None, noise=0):
        """Evaluates the gradient of the loss at point x wrt a."""
        if x is None:
            x = np.random.rand()
        return - 2 * (teacher(x) + noise - self(x)) * self.sigma(x - self.u)
    

    def stoch_grad_u_loss(self, teacher, x=None, noise=0):
        """Evaluates the gradient of the loss at point x wrt u."""
        if x is None:
            x = np.random.rand()
        indicator = np.all([x >= self.u - self.eta/2, x <= self.u + self.eta/2], axis=0)
        return 2 * (teacher(x) + noise - self(x)) * self.a * indicator / self.eta
