# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        extendedFeatureMatrix: numpy array of shape (N,d+1)"""

    extendedFeatureMatrix = np.zeros((len(x),degree+1))
    for d in range(degree+1):
        extendedFeatureMatrix[:,d] = x**d
    
    return extendedFeatureMatrix
