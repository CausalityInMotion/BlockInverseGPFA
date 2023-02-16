"""
GPFA preprocessing functions.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np


class EventTimesToCounts(object):
    """
    This class bins spike trains of time points and provides
    a method to covernt `neo.SpikeTrains` data to a matrix
    with counted time points

    Parameters
    ----------
        bin_size : int
            Data bin width in [s] e.g., 0.02, 0.05, 0.1, 1, etc.
            Default : 0.02 [s]
        t_stop : int
            The stop time of the trial. If not specified, it is
            retrieved from the `t_stop` attribute of `neo.Spiketrain`
            or defaults to the maximum value of `X[0].shape[0]`.
            Default : None
        extrapolate_last_bin : boolean
            Extrapolates the count of the last bin
            Default : False

    Method
    ------
    transform

    """
    def __init__(self, bin_size=0.02, t_stop=None,
                 extrapolate_last_bin=False):
        self.bin_size = bin_size
        self.t_stop = t_stop
        self.extrapolate_last_bin = extrapolate_last_bin

        # for computational convenience
        self.bin_size = int(self.bin_size * 1000)

    def transform(self, X):
        """
        Transforms data from event times to binned counts

        Parameter
        ---------
        X : numpy.array or neo.SpikeTrain
            An array-like of observation sequences from one trial.
            Each element in X must be of the same length.

        Returns
        -------
        X_out : numpy.array
            An array-like of observation sequences from one trial.
            Each element in `X_out` is a matrix of size
            #x_dim x #bins, containing binned observation sequence.
        """
        X_out = np.empty(len(X), object)

        for i, spiketrain in enumerate(X):
            if hasattr(spiketrain, 't_spot'):
                self.t_stop = int(spiketrain.t_stop.magnitude)
                spiketrain = spiketrain.magnitude
            else:
                if self.t_stop is None:
                    self.t_stop = spiketrain.shape[0]
                spiketrain = np.where(spiketrain)[0]

            t_start = 0
            edges = np.arange(t_start, self.t_stop, self.bin_size)
            binned_spikecounts = np.histogram(spiketrain, edges)[0]
            # extrapolate the last  bin
            if self.extrapolate_last_bin:
                scale = binned_spikecounts[-1] / (self.t_stop % self.bin_size)
                binned_spikecounts[-1] = np.ceil(scale * self.bin_size)
            X_out[i] = binned_spikecounts

        return X_out
