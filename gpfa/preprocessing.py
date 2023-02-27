"""
GPFA preprocessing Class for neural data.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:license: Modified BSD, see LICENSE.txt for details.
"""

import sklearn
import numpy as np


class EventTimesToCounts(sklearn.base.TransformerMixin):
    """
    This class bins spike trains of time points and provides
    a method to covernt `neo.SpikeTrains` data to a matrix
    with counted time points

    This below serves as notes for when we update the
    docstring, please ignore them
    ===========================================================
    # If `t_stop` is not specified by the user it is
    # handled in two ways depending on the data type of `X`. If `X`
    # is a `neo.SpikeTrain` object, `t_stop` is set by getting it
    # from `X[0].t_stop.magnitude` since it is assumed that `t_stop`
    # is the same across all neurons. However, if `X` is not a
    # `neo.SpikeTrain` object then `t_stop` is obtained by finding
    # the largest spike time across all neurons

    # we add a small fraction of the `bin_size` to `t_stop` to
    # ensure that the last bin is included whenever `t_stop` falls
    # on the boundary. Furthermore, this helps to include spikes
    # that fall be in the last bin.

    # If `True`, check if `t_stop` falls on the boundary of the last
    # bin. If it indeed falls on the boundary, then
    # `extrapolate_last_bin` is set to `False` for this particular
    # trial since extrapolating the last bin is only required when
    # `t_stop` falls within the last bin
    =============================================================

    Parameters
    ----------
    bin_size : float
        Data bin width in [s] e.g., 0.02, 0.05, 0.1, 1, etc.
        Default : 0.02 [s]
    t_stop : float
        The stop time of the trial. If not specified, it is
        retrieved from the `t_stop` attribute of `neo.Spiketrain`
        or defaults to the largest spike time across all neurons.
        Default : None
    extrapolate_last_bin : boolean
        Controls if the last bin includes t_stop.
        If `False`, the last bin is the one before t_stop (or includes
        t_stop if it matches the bin's final edge). Then, all events
        after the bin's final edge are ignored. In this case, X_out
        contains integer event counts.
        If 'True', the last bin includes t_stop, and the event counts
        are extrapolated to take into account that t_stop is smaller
        than the last bin's final edge. In this case, X_out contains
        float event counts which, for the final bin, might be
        non-integer values.
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

    def transform(self, X):
        """
        Transforms data from event times to binned counts

        Parameter
        ---------
        X : numpy.array or neo.SpikeTrain
            An array-like of observation sequences from one trial.
            Each element in `X` can be of different lengths but the
            trial duration (i.e., `t_stop`) is the same across all
            neuron.

        Returns
        -------
        X_out : numpy.array
            A numpy matrix of size #neurons x #bins, containing the
            final bin counts.
        """
        # ==============================================
        # set the starting time of the trial (`t_start`)
        # and the end time (`t_stop`)
        # ==============================================
        t_start = 0
        if self.t_stop is None:
            if hasattr(X[0], 't_stop'):
                self.t_stop = X[0].t_stop.magnitude
            else:
                self.t_stop = max(map(lambda x: x[-1], X))

        # ====================================
        # get the bins based on the `bin_size`
        # ====================================
        edges = np.arange(t_start,
                          self.t_stop + self.bin_size * 0.1,
                          self.bin_size)

        # =============================
        # Check edges for extrapolation
        # =============================
        # we do not want any edge beyond `t_stop`,
        # if there is any edge `> t_stop` we remove it
        if edges[-1] > self.t_stop:
            edges = edges[:-1]
        # Check if user wants to extrapolate the last bin
        if self.extrapolate_last_bin:
            if self.t_stop > edges[-1]:
                edges = np.hstack((edges, edges[-1] + self.bin_size))
                last_bin_scaling = self.bin_size / (self.t_stop - edges[-2])
            else:
                self.extrapolate_last_bin = False

        # =======================
        # create an output matrix
        # =======================
        X_out = np.empty((len(X), len(edges) - 1),
                         dtype=(float if self.extrapolate_last_bin else int))

        # ======================================
        # Loop over each neuron in a given trial
        # to get the binned spike counts
        # ======================================
        for i, spiketrain in enumerate(X):

            # If neo.SpikeTrain, get the timesteps
            # of each neuron via `spiketrain.magnitude`
            if hasattr(spiketrain, 'units'):
                if self.t_stop != spiketrain.t_stop.magnitude:
                    raise ValueError(
                        f'The specified or computed `t_stop`: {self.t_stop}'
                        f'is different from the {i}_th spikeTrain `t_stop`'
                        "`t_stop` must be the same across all neurons "
                    )
                spiketrain = spiketrain.magnitude

            # binning happens here
            X_out[i, :] = np.histogram(spiketrain, edges)[0]
        # ========================
        # extrapolate the last bin
        # ========================
        if self.extrapolate_last_bin:
            X_out[:, -1] *= last_bin_scaling

        return X_out
