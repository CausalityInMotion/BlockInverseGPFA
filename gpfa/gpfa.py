"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
:cite:`gpfa-Yu2008_1881` for neural trajectory visualization of parallel spike
trains. GPFA applies factor analysis (FA) to time-binned spike count data to
reduce the dimensionality and at the same time smoothes the resulting
low-dimensional trajectories by fitting a Gaussian process (GP) model to them.

The input consists of a set of trials (X), each containing a list of spike
trains (N neurons). The output is the projection (Y) of the data in a space
of pre-chosen dimensionality y_dim < N.

Under the assumption of a linear relation (transform matrix C) between the
latent variable Y following a Gaussian process and the spike train data X with
a bias d and  a noise term of zero mean and (co)variance R (i.e.,
:math:`X = C Y + d + Gauss(0,R)`), the projection corresponds to the
conditional probability E[Y|X].
The parameters (C, d, R) as well as the time scales and variances of the
Gaussian process are estimated from the data using an expectation-maximization
(EM) algorithm.

Externally, as part of the preprocessing:

i) bin the spike train data to get a sequence of N dimensional vectors of spike
counts in respective time bins, and choose the reduced dimensionality y_dim
(c.f., `gpfa_processing.get_seqs()`)

ii) remove any inactive units (c.f., `gpfa_processing.remove_inactive_units`)

Internally, the analysis consists of the following steps:

1) expectation-maximization for fitting of the parameters C, d, R and the
time-scales and variances of the Gaussian process, using all the trials
provided as input (c.f., `gpfa_core.em()`)

2) projection of single trials in the low dimensional space (c.f.,
`gpfa_core.exact_inference_with_ll()`)

3) orthonormalization of the matrix C and the corresponding subspace, for
visualization purposes: (c.f., `gpfa_core.orthonormalize()`)



Original code
-------------
The code was ported from the MATLAB code based on Byron Yu's implementation.
The original MATLAB code is available at Byron Yu's website:
https://users.ece.cmu.edu/~byronyu/software.shtml

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import sklearn
from . import gpfa_core


__all__ = [
    "GPFA"
]


class GPFA(sklearn.base.BaseEstimator):
    """
    Apply Gaussian process factor analysis (GPFA) to spike train data

    There are two principle scenarios of using the GPFA analysis, both of which
    can be performed in an instance of the GPFA() class.

    In the first scenario, only one single dataset is used to fit the model and
    to extract the neural trajectories. The parameters that describe the
    transformation are first extracted from the data using the `fit()` method
    of the GPFA class. Then the same data is projected into the orthonormal
    basis using the method `transform()`. The `fit_transform()` method can be
    used to perform these two steps at once.

    In the second scenario, a single dataset is split into training and test
    datasets. Here, the parameters are estimated from the training data. Then
    the test data is projected into the low-dimensional space previously
    obtained from the training data. This analysis is performed by executing
    first the `fit()` method on the training data, followed by the
    `transform()` method on the test dataset.

    The GPFA class is compatible to the cross-validation functions of
    `sklearn.model_selection`, such that users can perform cross-validation to
    search for a set of parameters yielding best performance using these
    functions.

    Parameters
    ----------
    y_dim : int, optional
        state dimensionality
        Default: 3
    bin_size : float, optional
        spike bin width in sec
        Default: 0.02
    min_var_frac : float, optional
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
        Default: 0.01
        (See Martin & McDonald, Psychometrika, Dec 1975.)
    em_tol : float, optional
        stopping criterion for EM
        Default: 1e-8
    em_max_iters : int, optional
        number of EM iterations to run
        Default: 500
    tau_init : float, optional
        GP timescale initialization in sec
        Default: 0.1
    eps_init : float, optional
        GP noise variance initialization
        Default: 1e-3
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Attributes
    ----------
    valid_data_names : tuple of str
        Names of the data contained in the resultant data structure, used to
        check the validity of users' request
    has_spikes_bool : np.ndarray of bool
        Indicates if a neuron has any spikes across trials of the training
        data.
    params_estimated : dict
        Estimated model parameters. Updated at each run of the fit() method.

        covType : str
            type of GP covariance, either 'rbf', 'tri', or 'logexp'.
            Currently, only 'rbf' is supported.
        gamma : (1, #latent_vars) np.ndarray
            related to GP timescales of latent variables before
            orthonormalization by :math:`bin_size / sqrt(gamma)`
        eps : (1, #latent_vars) np.ndarray
            GP noise variances
        d : (#units, 1) np.ndarray
            observation mean
        C : (#units, #latent_vars) np.ndarray
            loading matrix, representing the mapping between the observed data
            space and the latent variable space
        R : (#units, #latent_vars) np.ndarray
            observation noise covariance
    fit_info : dict
        Information of the fitting process. Updated at each run of the fit()
        method.

        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
        log_likelihoods : list
            log likelihoods after each EM iteration.
    transform_info : dict
        Information of the transforming process. Updated at each run of the
        transform() method.

        log_likelihood : float
            maximized likelihood of the transformed data
        num_bins : nd.array
            number of bins in each trial
        Corth : (#units, #latent_vars) np.ndarray
            mapping between the observed data space and the orthonormal
            latent variable space

    Methods
    -------
    fit
    transform
    fit_transform
    score

    Example
    --------
    The following example computes the trajectories sampled from a random
    multivariate Gaussian process.

    >>> import numpy as np
    >>> from gpfa import GPFA, gpfa_util

    >>> # set random parameters
    >>> seed = [0, 8, 10]
    >>> bin_size = 0.02                             # [s]
    >>> sigma_f = 1.0
    >>> sigma_n = 1e-8
    >>> tau_f = 0.7
    >>> num_trials = 3                              # number of trials
    >>> N = 10                                      # number of units
    >>> y_dims = 3                                  # number of latent state


    >>> # get some finte number of points
    >>> t = np.arange(0, 10, 0.01).reshape(-1,1)  # time series
    >>> timesteps = len(t)                        # number of time points

    >>>  X = []
    >>> for n in range(num_trials):
    >>>     np.random.seed(seed[n])
    >>>     C = np.random.uniform(0, 2, (N, y_dims))    # loading matrix
    >>>     obs_noise = np.random.uniform(0.2, 0.75, N) # rand noise parameters

    >>>     # mean
    >>>     mu = np.zeros(t.shape)
    >>>     # Create covariance matrix for GP using the squared
    >>>     # exponential kernel from Yu et al.
    >>>     sqdist = (t - t.T)**2
    >>>     cov = sigma_f**2 * np.exp(-0.5 / tau_f**2 * sqdist)
    ...                            + sigma_n**2 * np.eye(timesteps)

    >>>     # Draw three latent state samples from a Gaussian process
    >>>     # using the above cov
    >>>     Y = np.random.multivariate_normal(mu.ravel(), cov, y_dims)

    >>>     # observations have Gaussian noise
    >>>     x = C@Y + np.random.normal(0, obs_noise, (timesteps, N)).T
    >>>     X.append(x)

    >>> data = np.array(X)

    >>> gpfa = GPFA(bin_size=bin_size, x_dim=2)
    >>> gpfa.fit(data)
    >>> results = gpfa.transform(data, returned_data=['latent_variable_orth',
    ...                                               'latent_variable'])
    >>> latent_variable_orth = results['latent_variable_orth']
    >>> latent_variable = results['latent_variable']

    or simply

    >>> results = GPFA(bin_size=bin_size, x_dim=x_dims).fit_transform(data,
    ...                returned_data=['latent_variable_orth',
    ...                               'latent_variable'])
    """

    def __init__(self, bin_size=0.02, y_dim=3, min_var_frac=0.01,
                 tau_init=0.1, eps_init=1.0E-3, em_tol=1.0E-8,
                 em_max_iters=500, freq_ll=5, verbose=False):
        self.bin_size = bin_size
        self.y_dim = y_dim
        self.min_var_frac = min_var_frac
        self.tau_init = tau_init
        self.eps_init = eps_init
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names = (
            'latent_variable_orth',
            'latent_variable',
            'Vsm',
            'VsmGP',
            'x')
        self.has_spikes_bool = None
        self.verbose = verbose

        # will be updated later
        self.params_estimated = {}
        self.fit_info = {}
        self.transform_info = {}

    def fit(self, X):
        """
        Fit the model with the given training data.

        Parameters
        ----------
        X   : list
            training data structure containing np.ndarrays whose n-th
            element (corresponding to the n-th experimental trial) has
            shape of (#units, #bins)

        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError

            If covariance matrix of input data is rank deficient.
        """
        # Get the dimension of training data
        self.has_spikes_bool = np.hstack(X).any(axis=1)
        # Check if training data covariance is full rank
        x_all = np.hstack(X)
        x_dim = x_all.shape[0]

        if np.linalg.matrix_rank(np.cov(x_all)) < x_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if self.verbose:
            print(f'Number of training trials: {len(X)}')
            print(f'Latent space dimensionality: {self.y_dim}')
            print(f'Observation dimensionality: {self.has_spikes_bool.sum()}')

        # The following does the heavy lifting.
        self.params_estimated, self.fit_info = gpfa_core.fit(
            X=X,
            y_dim=self.y_dim,
            bin_size=self.bin_size,
            min_var_frac=self.min_var_frac,
            em_max_iters=self.em_max_iters,
            em_tol=self.em_tol,
            tau_init=self.tau_init,
            eps_init=self.eps_init,
            freq_ll=self.freq_ll,
            verbose=self.verbose)

        return self

    def transform(self, X, returned_data=None):
        """
        Obtain trajectories of neural activity in a low-dimensional latent
        variable space by inferring the posterior mean of the obtained GPFA
        model and applying an orthonormalization on the latent variable space.

        Parameters
        ----------
        X   : list
            training data structure containing np.ndarrays whose n-th
            element (corresponding to the n-th experimental trial) has
            shape of (#units, #bins)

        returned_data : list of str
            Set `returned_data` to a list of str of desired resultant data e.g:
            `returned_data = ['latent_variable_orth']`
            The dimensionality reduction transform generates the following
            resultant data:

               'latent_variable_orth': orthonormalized posterior mean of latent
               variable

               'latent_variable': posterior mean of latent variable before
               orthonormalization

               'Vsm': posterior covariance between latent variables

               'VsmGP': posterior covariance over time for each latent variable

               'x': observed data used to estimate the GPFA model parameters

            `returned_data` specifies the keys by which the data dict is
            returned.

            Default is ['latent_variable_orth'].

        Returns
        -------
        np.ndarray or dict
            When the length of `returned_data` is one, a single np.ndarray,
            containing the requested data (the first entry in `returned_data`
            keys list), is returned. Otherwise, a dict of multiple np.ndarrays
            with the keys identical to the data names in `returned_data` is
            returned.

            N-th entry of each np.ndarray is a np.ndarray of the following
            shape, specific to each data type, containing the corresponding
            data for the n-th trial:

                `latent_variable_orth`: (#latent_vars, #bins) np.ndarray

                `latent_variable`:  (#latent_vars, #bins) np.ndarray

                `x`:  (#units, #bins) np.ndarray

                `Vsm`:  (#latent_vars, #latent_vars, #bins) np.ndarray

                `VsmGP`:  (#bins, #bins, #latent_vars) np.ndarray

            Note that the num. of bins (#bins) can vary across trials,
            reflecting the trial durations in the given `observed` data.

        Raises
        ------
        ValueError
            If the number of units in `observations` is different from that
            in the training data.

            If `returned_data` contains keys different from the ones in
            `self.valid_data_names`.
        """
        if returned_data is None:
            returned_data = ['latent_variable_orth']

        if len(X[0]) != len(self.has_spikes_bool):
            raise ValueError("'observed data' must contain the same number of "
                             "units as the training data")
        invalid_keys = set(returned_data).difference(self.valid_data_names)
        if len(invalid_keys) > 0:
            raise ValueError("'returned_data' can only have the following "
                             f"entries: {self.valid_data_names}")

        seqs, ll = gpfa_core.exact_inference_with_ll(X,
                                                     self.params_estimated,
                                                     get_ll=True)
        self.transform_info['log_likelihood'] = ll
        self.transform_info['num_bins'] = [nb.shape[1] for nb in seqs['x']]
        Corth, seqs = gpfa_core.orthonormalize(self.params_estimated, seqs)
        self.transform_info['Corth'] = Corth
        if len(returned_data) == 1:
            return seqs[returned_data[0]]
        return {i: seqs[i] for i in returned_data}

    def fit_transform(self, X, returned_data=None):
        """
        Fit the model with `observed` data and apply the dimensionality
        reduction on the `observations`.

        Parameters
        ----------
        X   : list of observed data arrays per trial
            Refer to the :func:`GPFA.fit` docstring.

        returned_data : list of str
            Refer to the :func:`GPFA.transform` docstring.

        Returns
        -------
        np.ndarray or dict
            Refer to the :func:`GPFA.transform` docstring.

        Raises
        ------
        ValueError
             Refer to :func:`GPFA.fit` and :func:`GPFA.transform`.

        See Also
        --------
        GPFA.fit : fit the model with `observation`
        GPFA.transform : transform `obsrvation` into trajectories

        """
        if returned_data is None:
            returned_data = ['latent_variable_orth']
        self.fit(X)
        return self.transform(X, returned_data=returned_data)

    def score(self, X):
        """
        Returns the log-likelihood of the given data under the fitted model

        Parameters
        ----------
        X   : list
            training data structure containing np.ndarrays whose n-th
            element (corresponding to the n-th experimental trial) has
            shape of (#units, #bins)

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the given spiketrains under the fitted model.
        """
        self.transform(X)
        return self.transform_info['log_likelihood']
