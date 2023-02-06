"""
GPFA preprocessing functions.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def transformer(X, bin_data=True, bin_size=0.02, transform_func=None):
    """
    Transforms data from an arbitrary callable, and converts binary
    data into a numpy.ndarray of binned counts.

    Parameters
    ----------
    X   : an array-like of observation sequences, one per trial.
        Each element in X is a matrix of size #x_dim x #bins,
        containing an observation sequence. The input dimensionality
        #x_dim needs to be the same across elements in X, but #bins
        can be different for each observation sequence.
        Default : None
    bin_data: boolean
        Bin data into the specificied bin width (i.e., `bin_size`)
        Default : True
    bin_size: int
        Data bin width in [s] e.g., 0.02, 0.05, 0.1, 1, etc.
        Default : 0.02 [s]
    transform_func : callable
        The callable function to use in data transformation.
        Default: None

    Returns
    -------
    X_out : numpy.ndarray.
        An array-like of observation sequences, one per trial.
        Has same data structure as `X` but may have less #x_dim
        if any rows of the X[n] contained all zeros. If the data
        is binned, each element in `X_out` is a matrix of size
        #x_dim x #bins, containing an observation sequence within
        # each trial. The input dimensionality `x_dim` needs to be
        # the same across all elements in `X`, but #bins can differ.
    """
    # ====================================
    # Remove inactive rows from the data
    # ====================================
    X_out = X
    # Get the dimension of training data
    non_zero_x_dim = np.hstack(X_out).any(axis=1)
    for X_out_n in X_out:
        X_out_n = X_out_n[non_zero_x_dim, :]

    # for computational convenience change bin_size to int
    bin_size = int(bin_size * 1000)

    # loop over all trials
    for n, X_n in enumerate(X_out):
        # ====================
        # Bin data
        # ====================
        if bin_data:
            # number of units
            x_dim = X_n.shape[0]
            # number of bins per trial
            n_bin = int(np.floor(X_n.shape[1]/bin_size))
            binned_spikecount = np.zeros([x_dim, n_bin])

            # loop over the number of bins to compute the
            # count per bin
            for b in range(n_bin):

                # bin egdes
                t_start = bin_size * b
                t_stop = bin_size * (b + 1)
                binned_spikecount[:, b] = np.sum(
                                        X_n[:, t_start:t_stop],
                                        axis=1
                                        )
        # ============================
        # Transform data
        # ============================
        if transform_func is not None:
            if bin_data:
                transform = FunctionTransformer(transform_func)
                X_transformed = transform.transform(binned_spikecount)
            else:
                transform = FunctionTransformer(transform_func)
                X_transformed = transform.transform(X_n)

        X_out[n] = X_transformed

    # =================================================
    # Remove trials that are shorter than one bin width
    # =================================================
    if len(X_out) > 0:
        trials_to_keep = np.array([X_out_n.shape[1] for X_out_n in X_out]) > 0
        X_out = np.array(X_out)[(trials_to_keep)]
    return X_out