# ...
# Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
# license Modified BSD, see LICENSE.txt for details.
# ...

from __future__ import division, print_function, unicode_literals

import sklearn
import numpy as np

__all__ = [
    "EventTimesToCounts"
]


class EventTimesToCounts(sklearn.base.TransformerMixin):
    """
    Bins sequence of event times into event counts whithin evenly spaced
    time bins.

    This class supports binning sequences of event times (e.g., spike trains)
    into a matrix that contain event counts within evenly spaced time bins
    (first bin starts at time 0s). It supports specifying the bin size, and
    multiple options for how the last event time is determined.

    Parameters
    ----------
    bin_size : float, default : 0.02 [s]
        The width of each time bin in seconds.

    t_stop : float, optional, default : None
        The largest time considered for binning. This time is assumed to
        be the same across all event time sequences. If `t_stop` doesn't match
        a bin boundary, `extrapolate_last_bin` determins whether or not the
        last bin includes `t_stop` or not.

        If not given, `t_stop` is set to the largest event time across all
        sequences in the provided data. If the data is a `neo.SpikeTrain`
        object, `t_stop` is set to `X[0].t_stop.magnitude`.

    extrapolate_last_bin : boolean, optional, default : False
        In cases where `t_stop` does not match a bin boundary, this option
        determines whether the last bin includes `t_stop`. If `False`, the last
        bin ends before `t_stop`, and all events after this final bin are
        ignored. If `True`, the last bin includes `t_stop`, and event counts
        are up-scaled to account for the fact that `t_stop` happens before the
        end of the last time bin. For example, if `t_stop` falls into the
        middle of the last bin, all event counts that fall into the last bin
        are doubled. In this case, `X_out` of :meth:`transform` returns is of
        type `float` as it might contain non-integer event counts for the last
        bin.

    Examples
    --------
    >>> import numpy as np
    >>> from gpfa.preprocessing import EventTimesToCounts
    >>> bin_size = 0.1  # [s]
    >>> t_stop = 0.8  # [s]
    >>> X = [
    ...     [0, 0.1, 0.15, 0.4, 0.5, 0.6, 0.8],
    ...     [0.05, 0.3, 0.4, 0.55, 0.7]
    ...     ]
    >>> ettc = EventTimesToCounts(
    ...                           bin_size=bin_size,
    ...                           t_stop=t_stop,
    ...                           extrapolate_last_bin=False
    ...                          )
    >>> ettc.transform(X)
    array([[1, 2, 0, 0, 1, 2, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0]])

    >>> ettc_extrapolate_last_bin = EventTimesToCounts(
    ...                             bin_size=bin_size,
    ...                             t_stop=t_stop,
    ...                             extrapolate_last_bin=True
    ...                          )
    >>> ettc.transform(X)
    array([[1, 2, 0, 0, 1, 2, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0]])

    >>> # Using defaults t_stop=None and extrapolate_last_bin=False
    >>> ettc = EventTimesToCounts(bin_size)
    >>> ettc.transform(X)
    array([[1, 2, 0, 0, 1, 2, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0]])

    >>> t_stop2 = 0.88  # [s]
    >>> ettc = EventTimesToCounts(
    ...                           bin_size=bin_size,
    ...                           t_stop=t_stop2,
    ...                           extrapolate_last_bin=False
    ...                          )
    >>> ettc.transform(X)
    array([[1, 2, 0, 0, 1, 2, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0]])

    >>> t_stop2 = 0.88  # [s]
    >>> ettc_extrapolate_last_bin = EventTimesToCounts(
    ...                             bin_size=bin_size,
    ...                             t_stop=t_stop2,
    ...                             extrapolate_last_bin=True
    ...                          )
    >>> ettc.transform(X)
    array([[1.  , 2.  , 0.  , 0.  , 1.  , 2.  , 0.  , 0.  , 1.25],
           [1.  , 0.  , 1.  , 0.  , 1.  , 1.  , 1.  , 0.  , 0.  ]])

    The following example only works if the `Neo package
    <https://neo.readthedocs.io/>`_ is installed.

    >>> import neo
    >>> t_stop = 0.8  # [s]
    >>> neoSpikeTrain = [
    ...     neo.SpikeTrain(X[0],units='sec', t_stop=t_stop),
    ...     neo.SpikeTrain(X[1], units='sec', t_stop=t_stop)
    ...     ]
    >>> ettc = EventTimesToCounts(
                        bin_size=bin_size,
                        t_stop=None,
                        extrapolate_last_bin=False
                        )
    >>> ettc.transform(neoSpikeTrain)
    array([[1, 2, 0, 0, 1, 2, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0]])

    Methods
    -------
    transform:
        Transforms data from event times to binned event counts

    """
    def __init__(self, bin_size=0.02, t_stop=None,
                 extrapolate_last_bin=False):
        self.bin_size = bin_size
        self.t_stop = t_stop
        self.extrapolate_last_bin = extrapolate_last_bin

    def transform(self, X):
        """
        Transforms data from event times to binned event counts

        Parameters
        ----------
        X : numpy.array or neo.SpikeTrain
            An array-like containing #sequences of event time sequences
            (usually sequences of `float`'s). Each element in `X` can contain a
            different number of event times. The are all assumed to share the
            same final time (i.e., ``t_stop``).

        Returns
        -------
        X_out : numpy.array
            A numpy matrix of size #sequences x #bins, containing the
            binned event counts.
        """
        # ==========================================================================
        # set the starting time of the trial (`t_start`) and the end time (`t_stop`)
        # ==========================================================================
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

        # =============================================================
        # Loop over event time sequences to compute binned spike counts
        # =============================================================
        for i, eventseq in enumerate(X):

            # If neo.SpikeTrain, get the timesteps
            # of each neuron via `eventseq.magnitude`
            if hasattr(eventseq, 'units'):
                if self.t_stop != eventseq.t_stop.magnitude:
                    raise ValueError(
                        f'The specified or computed `t_stop`: {self.t_stop} '
                        f'is different from the {i}_th spikeTrain `t_stop` '
                        "`t_stop` must be the same across all neurons."
                    )
                eventseq = eventseq.magnitude

            # binning happens here
            X_out[i, :] = np.histogram(eventseq, edges)[0]
        # ========================
        # extrapolate the last bin
        # ========================
        if self.extrapolate_last_bin:
            X_out[:, -1] *= last_bin_scaling

        return X_out
