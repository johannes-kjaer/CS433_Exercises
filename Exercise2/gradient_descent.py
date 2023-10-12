# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
from costs import compute_loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w for the mean square error .
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        w: (d,) array of model parameters/weights
    Returns:
        (d,) array containing the gradient at w
    """
    return - tx.T @ (y-tx @ w) / len(y)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        max_iters: Integer denoting the maximum number of iterations of GD
        gamma: Float denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = np.zeros((max_iters+1,len(initial_w)))
    ws[0] = initial_w
    losses = np.zeros(max_iters)

    for n in range(max_iters):
        losses[n] = compute_loss(y,tx,ws[n])
        grad_n = compute_gradient(y,tx,ws[n])
        
        ws[n+1] = ws[n] - grad_n * gamma

        print(f"GD iter. {n+1}/{max_iters}: loss={losses[n]}, w={ws[n]}")

    return losses, ws
