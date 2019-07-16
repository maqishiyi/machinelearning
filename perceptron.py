import numpy as np
class Perceptron(object) :
    """
    Perceptron classifier

    Parameters
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.

    Attributes
    w_: ld-array
        Weights after fitting
    errors_: list
        Number of misclassifications in every epoch
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter


