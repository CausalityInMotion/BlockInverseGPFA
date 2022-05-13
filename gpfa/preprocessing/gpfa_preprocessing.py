"""
GPFA preprocessing functions.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

import numpy as np


def remove_inactive_units(X):
    """
    Remove inactive units based on training set
    """
    seqs = X
    has_spikes_bool = np.hstack(seqs).any(axis=1)
    for seq in seqs:
        seq = seq[has_spikes_bool, :]
    return seqs


def get_seqs(data, bin_size=0.02, use_sqrt=True):
    """
    Converts binary spike trains into a rec array of spike counts.

    Parameters
    ----------
    data : A list of numpy.ndarray
        data structure containing np.ndarrays whose n-th element
        (corresponding to the n-th experimental trial) has shape
        of (#units, #bins).
    bin_size: int
        Spike bin width in [s] e.g., 0.02, 0.05, 0.1, 1, etc.
        Default: 0.02 [s]
    use_sqrt: bool
        Boolean specifying whether or not to use square-root transform on
        spike counts.
        Default: True

    Returns
    -------
    seq : np.recarray
        data structure, whose nth entry (corresponding to the nth experimental
        trial) has shape of (#units, #bins)

    Raises
    ------
    TypeError
        if `data` type is not a list.
        if `data` type is not a list containg np.ndarrays.
    """

    if not isinstance(data, list):
        raise TypeError("'data' must be a 'list'")
    for d in data:
        if not isinstance(d, np.ndarray):
            raise TypeError("'data' must be a 'list' containing np.ndarrays")

    seqs = []

    # for computational convenience change bin_size to int
    bin_size = int(bin_size * 1000)

    # loop over all trials
    for t, seq in enumerate(data):

        # number of units
        xdim = len(seq)
        # number of bins per trial
        n_bin = int(np.floor(data[0].shape[1]/bin_size))
        binned_spikecount = np.zeros([xdim, n_bin])

        # loop over the number of bins to compute the
        # spike count per bin
        for b in range(n_bin):

            # bin egdes
            t_start = bin_size * b
            t_stop = bin_size * (b + 1)
            binned_spikecount[:, b] = np.sum(
                                    data[t][:, t_start:t_stop],
                                    axis=1
                                    )
        # take square root of the binned_spikeCount
        # if `use_sqrt` is True (see paper for motivation)
        if use_sqrt:
            binned_spikecount = np.sqrt(binned_spikecount)

        seqs.append(binned_spikecount)
    seqs = np.array(seqs)
    # Remove trials that are shorter than one bin width
    if len(seqs) > 0:
        trials_to_keep = np.array([nb.shape[1] for nb in seqs]) > 0
        seqs = seqs[trials_to_keep]
    return seqs
