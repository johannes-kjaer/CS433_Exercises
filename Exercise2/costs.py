# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def MSE(e):
    '''
    Calculates the mean squared error of the submitted error-array
    Args:
        e: (N,) array of the error fr all N predictions
    Returns:
        Float value of the mean squared error'''
    return e.T @ e / len(e)

def MAE(e):
    '''
    Calculates the mean absolute error of the submitted error-array
    Args:
        e: (N,) array of the error fr all N predictions
    Returns:
        Float value of the mean absolute error'''
    return np.sum(np.abs(e)) / len(e)

def compute_loss(y, tx, w, lossFunction=MSE):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        lossFunction: The loss function of your choice. Defaults to mean square error

    Returns:
        Float value of the loss, given the selected loss-function
    """
    return lossFunction(y - tx @ w)