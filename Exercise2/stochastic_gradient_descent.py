# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
import numpy as np
from costs import compute_loss

def compute_stoch_gradient(y,tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.
    Args:
        y: (B,) array of labels
        tx: (B,d) array of samples and their features
        w: (d,) array of model parameters
    Returns:
        (2,) array containing a stochastic gradient at w
    """
    return - tx.T @ (y - tx @ w) / len(y)

def mini_batch(y,tx,B):
    '''Extract B random labels and their corresponding samples
    Args: 
        y: (N,) array of labels
        tx: (N,d) array of samples and their features
        B: Integer denoting the desired batch size'''
    
    shuffledIndexes = np.random.permutation(len(y)) # Produces an array of lenght N with the indices 0 to N-1 in a random permutation
    return y[shuffledIndexes[0:B]] , tx[shuffledIndexes[0:B]] # Returns the samples of y and tx corresponding to the first B indices in our randomly permuted index array

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: (N,) array with the labels
        tx: (N,d) with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        batch_size: Integer denoting the desired number of data points to use for computing the stochastic gradient
        max_iters: Integer denoting the maximum number of iterations of SGD
        gamma: Float denoting the stepsize
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    # Define parameters to store w and loss
    ws = np.zeros((max_iters+1,len(initial_w)))
    ws[0] = initial_w
    losses = np.zeros(max_iters)

    for n in range(max_iters):
        losses[n] = compute_loss(y,tx,ws[n])

        batch_y, batch_x = mini_batch(y,tx,batch_size)
        ws[n+1] = ws[n] - gamma * compute_stoch_gradient(batch_y,batch_x,ws[n])

        print(f"SGD iter. {n+1}/{max_iters}: loss={losses[n]}, w={ws[n]}")

    return losses, ws