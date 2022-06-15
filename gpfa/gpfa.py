"""
Gaussian-process factor analysis (GPFA) is a dimensionality reduction method
:cite:`gpfa-Yu2008_1881` for latent trajectory visualization. GPFA applies f
actor analysis (FA) to observed data to reduce the dimensionality and at the
same time smoothes the resulting low-dimensional trajectories by fitting a
Gaussian process (GP) model to them.

The input consists of a set of trials (X), each containing a list of
observation sequences, one per trial. The output is the projection (Z) of the
data in a space of pre-chosen dimensionality z_dim < N.

Under the assumption of a linear relation (transform matrix C) between the
latent variable Z following a Gaussian process and the observation X with
a bias d and  a noise term of zero mean and (co)variance R (i.e.,
:math:`X = C@Z + d + Gauss(0,R)`), the projection corresponds to the
conditional probability E[Z|X].
The parameters (C, d, R) as well as the time scales and variances of the
Gaussian process are estimated from the data using an expectation-maximization
(EM) algorithm.

1) expectation-maximization for fitting of the parameters C, d, R and the
time-scales and variances of the Gaussian process, using all the trials
provided as input

2) projection of single trials in the low dimensional space

3) orthonormalization of the matrix C and the corresponding subspace, for
visualization purposes



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

import time
import warnings

import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize
import scipy.sparse as sparse
from tqdm import trange
import sklearn
import scipy as sp
from sklearn.utils.extmath import fast_logdet
from sklearn.decomposition import FactorAnalysis


__all__ = [
    "GPFA"
]


class GPFA(sklearn.base.BaseEstimator):
    """
    Apply Gaussian process factor analysis (GPFA) to observed data

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
    z_dim : int, optional
        latent state dimensionality
        Default: 3
    bin_size : float, optional
        observed data bin width in sec
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
    params_estimated : dict
        Estimated model parameters. Updated at each run of the fit() method.

        covType : str
            type of GP covariance, either 'rbf', 'tri', or 'logexp'.
            Currently, only 'rbf' is supported.
        gamma : (1, #z_dim) numpy.ndarray
            related to GP timescales of latent variables before
            orthonormalization by :math:`bin_size / sqrt(gamma)`
        eps : (1, #z_dim) numpy.ndarray
            GP noise variances
        d : (#x_dim, 1) numpy.ndarray
            observation mean
        C : (#x_dim, #z_dim) numpy.ndarray
            loading matrix, representing the mapping between the observed data
            space and the latent variable space
        R : (#x_dim, #z_dim) numpy.ndarray
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
        Corth : (#x_dim, #z_dim) numpy.ndarray
            mapping between the observed data space and the orthonormal
            latent variable space

    Methods
    -------
    _fit
    _transform
    _fit_transform
    _score

    Example
    --------
    The following example computes the trajectories sampled from a random
    multivariate Gaussian process.

    >>> import numpy as np
    >>> from gpfa import GPFA

    >>> # set random parameters
    >>> seed = [0, 8, 10]
    >>> bin_size = 0.02                             # [s]
    >>> sigma_f = 1.0
    >>> sigma_n = 1e-8
    >>> tau_f = 0.7
    >>> num_trials = 3                              # number of trials
    >>> N = 10                                      # number of units
    >>> z_dim = 3                                   # number of latent state

    >>> # get some finte number of points
    >>> t = np.arange(0, 10, 0.01).reshape(-1,1)  # time series
    >>> timesteps = len(t)                        # number of time points

    >>> C = np.random.uniform(0, 2, (N, z_dim))     # loading matrix
    >>> obs_noise = np.random.uniform(0.2, 0.75, N) # rand noise parameters

    >>> # mean
    >>> mu = np.zeros(t.shape)
    >>> # Create covariance matrix for GP using the squared
    >>> # exponential kernel from Yu et al.
    >>> sqdist = (t - t.T)**2
    >>> cov = sigma_f**2 * np.exp(-0.5 / tau_f**2 * sqdist)
    ...                         + sigma_n**2 * np.eye(timesteps)

    >>> X = []
    >>> for n in range(num_trials):
    >>>     np.random.seed(seed[n])

    >>>     # Draw three latent state samples from a Gaussian process
    >>>     # using the above cov
    >>>     Z = np.random.multivariate_normal(mu.ravel(), cov, z_dim)

    >>>     # observations have Gaussian noise
    >>>     x = C@Z + np.random.normal(0, obs_noise, (timesteps, N)).T
    >>>     X.append(x)

    >>> gpfa = GPFA(bin_size=bin_size, z_dim=2)
    >>> gpfa.fit(X)
    >>> results = gpfa.transform(data, returned_data=['pZ_mu_orth',
    ...                                               'pZ_mu'])
    >>> pZ_mu_orth = results['pZ_mu_orth']
    >>> pZ_mu = results['pZ_mu']

    or simply

    >>> results = GPFA(bin_size=bin_size, z_dim=z_dim).fit_transform(X,
    ...                returned_data=['pZ_mu_orth', 'pZ_mu'])
    """

    def __init__(self, bin_size=0.02, z_dim=3,
                 min_var_frac=0.01, tau_init=0.1, eps_init=1.0E-3,
                 em_tol=1.0E-8, em_max_iters=500, freq_ll=5, verbose=False):
        self.bin_size = bin_size
        self.z_dim = z_dim
        self.min_var_frac = min_var_frac
        self.tau_init = tau_init
        self.eps_init = eps_init
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names = (
            'pZ_mu_orth',
            'pZ_mu',
            'pZ_cov',
            'pZ_covGP',
            'X')
        self.verbose = verbose

        # will be updated later
        self.params_estimated = {}
        self.fit_info = {}
        self.transform_info = {}

    def fit(self, X, use_cut_trials=False):
        """
        Fit the model with the given training data.

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None
        use_cut_trials : bool, optional
            `use_cut_trials=True` results in using an approximation that might
            make the fitting computations more efficient. In most cases, this
            approximation shouldn't impact the fits qualitatively. It might do
            so if the data is expected to have very slow (i.e., long timescale)
            latent fluctuations.
            Default: False
        Returns
        -------
        self : object
            Returns the instance itself.

        Raises
        ------
        ValueError

            If covariance matrix of input data is rank deficient.
        """
        if use_cut_trials:
            # For compute efficiency, train on shorter segments of trials
            X = self._cut_trials(X)
            if len(X) == 0:
                warnings.warn('No segments extracted for training. Defaulting '
                              'to segLength=Inf.')
                X = self._cut_trials(X, seg_length=np.inf)
        # Check if training data covariance is full rank
        X_all = np.hstack(X)
        x_dim = X_all.shape[0]

        if np.linalg.matrix_rank(np.cov(X_all)) < x_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if self.verbose:
            print(f'Number of training trials: {len(X)}')
            print(f'Latent space dimensionality: {self.z_dim}')
            print(f'Observation dimensionality: {X_all.any(axis=1).sum()}')

        # The following does the heavy lifting.
        self.params_estimated, self.fit_info = self._fitting_core(
            X=X,
            z_dim=self.z_dim,
            bin_size=self.bin_size,
            min_var_frac=self.min_var_frac,
            em_max_iters=self.em_max_iters,
            em_tol=self.em_tol,
            tau_init=self.tau_init,
            eps_init=self.eps_init,
            freq_ll=self.freq_ll,
            verbose=self.verbose)

        return self

    def transform(self, X, returned_data=['pZ_mu_orth']):
        """
        Obtain trajectories of neural activity in a low-dimensional latent
        variable space by inferring the posterior mean of the obtained GPFA
        model and applying an orthonormalization on the latent variable space.

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None

        returned_data : list of str
            Set `returned_data` to a list of str of desired resultant data e.g:
            `returned_data = ['pZ_mu_orth']`
            The dimensionality reduction transform generates the following
            resultant data:

               'pZ_mu': posterior mean of latent variable before
               orthonormalization

               'pZ_mu_orth': orthonormalized posterior mean of latent
               variable

               'pZ_cov': posterior covariance between latent variables

               'pZ_covGP': posterior covariance over time for each latent 
                variable

               'X': observed data used to estimate the GPFA model parameters

            `returned_data` specifies the keys by which the data dict is
            returned.

            Default is ['pZ_mu_orth'].

        Returns
        -------
        numpy.ndarray or dict
            When the length of `returned_data` is one, a single numpy.ndarray,
            containing the requested data (the first entry in `returned_data` 
            keys list), is returned. Otherwise, a dict of multiple
            numpy.ndarrays with the keys identical to the data names in
            `returned_data` is returned.

            N-th entry of each numpy.ndarray is a numpy.ndarray of the
            following shape, specific to each data type, containing the
            corresponding data for the n-th trial:

                `pZ_mu`:  (#z_dim, #bins) numpy.ndarray

                `pZ_mu_orth`: (#z_dim, #bins) numpy.ndarray

                `X`:  (#x_dim, #bins) numpy.ndarray

                `pZ_cov`:  (#z_dim, #z_dim, #bins) numpy.ndarray

                `pZ_covGP`:  (#bins, #bins, #z_dim) numpy.ndarray

            Note that the num. of bins (#bins) can vary across trials,
            reflecting the trial durations in the given `observed` data.

        Raises
        ------
        ValueError
            If `returned_data` contains keys different from the ones in
            `self.valid_data_names`.
        """
        invalid_keys = set(returned_data).difference(self.valid_data_names)
        if len(invalid_keys) > 0:
            raise ValueError("'returned_data' can only have the following "
                             f"entries: {self.valid_data_names}")

        seqs, ll = self._infer_latents(X, self.params_estimated, 
                                       get_ll=True)
        self.transform_info['log_likelihood'] = ll
        self.transform_info['num_bins'] = [nb.shape[1] for nb in seqs['X']]
        Corth, seqs = self._orthonormalize(self.params_estimated, seqs)
        self.transform_info['Corth'] = Corth
        if len(returned_data) == 1:
            return seqs[returned_data[0]]
        return {i: seqs[i] for i in returned_data}

    def fit_transform(self, X, returned_data=['pZ_mu_orth']):
        """
        Fit the model with `observed` data and apply the dimensionality
        reduction on the `observations`.

        Parameters
        ----------
        X   : an array-like of observed data arrays per trial
            Refer to the :func:`GPFA.fit` docstring.
        returned_data : list of str
            Refer to the :func:`GPFA.transform` docstring.

        Returns
        -------
        numpy.ndarray or dict
            Refer to the :func:`GPFA.transform` docstring.

        Raises
        ------
        ValueError
             Refer to :func:`GPFA.fit` and :func:`GPFA.transform`.

        See Also
        --------
        GPFA.fit : fit the model with `observation`
        GPFA.transform : transform `observation` into trajectories
        """
        self.fit(X)
        return self.transform(X, returned_data=returned_data)

    def score(self, X):
        """
        Returns the log-likelihood of the given data under the fitted model

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the given spiketrains under the fitted model.
        """
        self.transform(X)
        return self.transform_info['log_likelihood']

    def _fitting_core(self, X, z_dim=3, bin_size=0.02, min_var_frac=0.01,
                      em_tol=1.0E-8, em_max_iters=500, tau_init=0.1,
                      eps_init=1.0E-3, freq_ll=5, verbose=False):
        """
        Fit the GPFA model with the given training data.

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None
        z_dim : int, optional
            latent state dimensionality
            Default: 3
        bin_size : float, optional
            observed data bin width in sec
            Default: 0.02 [s]
        min_var_frac : float, optional
            fraction of overall data variance for each observed dimension to
            set as the private variance floor.  This is used to combat
            Heywood cases, where ML parameter learning returns one or more
            zero private variances.
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
            Default: 0.1 [s]
        eps_init : float, optional
            GP noise variance initialization
            Default: 1e-3
        freq_ll : int, optional
            data likelihood is computed at every freq_ll EM iterations.
            freq_ll = 1 means that data likelihood is computed at every
            iteration.
            Default: 5
        verbose : bool, optional
            specifies whether to display status messages
            Default: False

        Returns
        -------
        parameter_estimates : dict
            Estimated model parameters.
            When the GPFA method is used, following parameters are contained
                covType: {'rbf', 'tri', 'logexp'}
                    type of GP covariance
                gamma: numpy.ndarray of shape (1, #z_dim)
                    related to GP timescales by 'bin_size / sqrt(gamma)'
                eps: numpy.ndarray of shape (1, #z_dim)
                    GP noise variances
                d: numpy.ndarray of shape (#x_dim, 1)
                    observation mean
                C: numpy.ndarray of shape (#x_dim, #z_dim)
                    mapping between the observed data space and the
                    latent variable space
                R: numpy.ndarray of shape (#x_dim, #z_dim)
                    observation noise covariance

        fit_info : dict
            Information of the fitting process and the parameters used there
            iteration_time : list
                containing the runtime for each iteration step in the EM
                algorithm.
        """
        # ==================================
        # Initialize state model parameters
        # ==================================
        params_init = {}
        params_init['covType'] = 'rbf'
        # GP timescale
        # Assume binWidth is the time step size.
        params_init['gamma'] = (bin_size / tau_init) ** 2 * np.ones(z_dim)
        # GP noise variance
        params_init['eps'] = eps_init * np.ones(z_dim)

        # ========================================
        # Initialize observation model parameters
        # ========================================
        print('Initializing parameters using factor analysis...')

        X_all = np.hstack(X)
        fa = FactorAnalysis(
                        n_components=z_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(X_all, bias=True))
                        )
        fa.fit(X_all.T)
        params_init['d'] = X_all.mean(axis=1)
        params_init['C'] = fa.components_.T
        params_init['R'] = np.diag(fa.noise_variance_)

        # Define parameter constraints
        params_init['notes'] = {
            'learnKernelParams': True,
            'learnGPNoise': False,
            'RforceDiagonal': True,
        }

        # =====================
        # Fit model parameters
        # =====================
        print('\nFitting GPFA model...')

        params_est, X, ll_cut, iter_time = self._em(
            params_init, X, min_var_frac=min_var_frac,
            max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll,
            verbose=verbose
            )

        fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

        return params_est, fit_info

    def _em(self, params_init, X, max_iters=500, tol=1.0E-8, min_var_frac=0.01,
           freq_ll=5, verbose=False):
        """
        Fits GPFA model parameters using expectation-maximization (EM)
        algorithm.

        Parameters
        ----------
        params_init : dict
            GPFA model parameters at which EM algorithm is initialized
            covType : {'rbf', 'tri', 'logexp'}
                type of GP covariance
            gamma : numpy.ndarray of shape (1, #z_dim)
                related to GP timescales by
                'bin_size / sqrt(gamma)'
            eps : numpy.ndarray of shape (1, #z_dim)
                GP noise variances
            d : numpy.ndarray of shape (#x_dim, 1)
                observation mean
            C : numpy.ndarray of shape (#x_dim, #z_dim)
                mapping between the observation data space and the
                latent variable space
            R : numpy.ndarray of shape (#x_dim, #z_dim)
                observation noise covariance
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None
        max_iters : int, optional
            number of EM iterations to run
            Default: 500
        tol : float, optional
            stopping criterion for EM
            Default: 1e-8
        min_var_frac : float, optional
            fraction of overall data variance for each observed dimension to
            set as the private variance floor.  This is used to combat Heywood
            cases, where ML parameter learning returns one or more zero
            private variances.
            Default: 0.01
            (See Martin & McDonald, Psychometrika, Dec 1975.)
        freq_ll : int, optional
            data likelihood is computed at every freq_ll EM iterations.
            freq_ll = 1 means that data likelihood is computed at every
            iteration.
            Default: 5
        verbose : bool, optional
            specifies whether to display status messages
            Default: False

        Returns
        -------
        params_est : dict
            GPFA model parameter estimates, returned by EM algorithm (same
            format as params_init)
        latent_seqs : numpy.recarray
            a copy of the training data structure, augmented with the new
            fields:
            pZ_mu : numpy.ndarray of shape (#z_dim x #bins)
                posterior mean of latent variables at each time bin
            pZ_cov : numpy.ndarray of shape (#z_dim, #z_dim, #bins)
                posterior covariance between latent variables at each
                timepoint
            pZ_covGP : numpy.ndarray of shape (#bins, #bins, #z_dim)
                posterior covariance over time for each latent
                variable
        ll : list
            list of log likelihoods after each EM iteration
        iter_time : list
            lisf of computation times (in seconds) for each EM iteration
        """
        params = params_init
        T = np.array([X_n.shape[1] for X_n in X])
        _, z_dim = params['C'].shape
        lls = []
        ll_old = ll_base = ll = 0.0
        iter_time = []
        var_floor = min_var_frac * np.diag(np.cov(np.hstack(X)))

        # Loop once for each iteration of EM algorithm
        for iter_id in trange(1, max_iters + 1, desc='EM iteration',
                              disable=not verbose):
            if verbose:
                print()
            tic = time.time()
            get_ll = (np.fmod(iter_id, freq_ll) == 0) or (iter_id <= 2)

            # ==== E STEP =====
            if not np.isnan(ll):
                ll_old = ll
            latent_seqs, ll = self._infer_latents(X, params, get_ll=get_ll)
            lls.append(ll)

            # ==== M STEP ====
            sum_p_auto = np.zeros((z_dim, z_dim))
            for seq_latent in latent_seqs:
                sum_p_auto += seq_latent['pZ_cov'].sum(axis=2) \
                    + seq_latent['pZ_mu'].dot(
                    seq_latent['pZ_mu'].T)
            X_all = np.hstack(X)
            Z_all = np.hstack(latent_seqs['pZ_mu'])
            sum_XZtrans = X_all.dot(Z_all.T)
            sum_Zall = Z_all.sum(axis=1)[:, np.newaxis]
            sum_Xall = X_all.sum(axis=1)[:, np.newaxis]

            # term is (z_dim+1) x (z_dim+1)
            term = np.vstack([np.hstack([sum_p_auto, sum_Zall]),
                            np.hstack([sum_Zall.T, T.sum().reshape((1, 1))])])
            # x_dim x (z_dim+1)
            cd = np.linalg.solve(term.T, np.hstack([sum_XZtrans, sum_Xall]).T).T

            params['C'] = cd[:, :z_dim]
            params['d'] = cd[:, -1]

            # yCent must be based on the new d
            # yCent = bsxfun(@minus, [seq.y], currentParams.d);
            # R = (yCent * yCent' - (yCent * [seq.Z_all]') * \
            #     currentParams.C') / sum(T);
            c = params['C']
            d = params['d'][:, np.newaxis]
            if params['notes']['RforceDiagonal']:
                sum_XXtrans = (X_all * X_all).sum(axis=1)[:, np.newaxis]
                xd = sum_Xall * d
                term = ((sum_XZtrans - d.dot(sum_Zall.T)) * c).sum(axis=1)
                term = term[:, np.newaxis]
                r = d ** 2 + (sum_XXtrans - 2 * xd - term) / T.sum()

                # Set minimum private variance
                r = np.maximum(var_floor, r)
                params['R'] = np.diag(r[:, 0])
            else:
                sum_XXtrans = X_all.dot(X_all.T)
                xd = sum_Xall.dot(d.T)
                term = (sum_XZtrans - d.dot(sum_Zall.T)).dot(c.T)
                r = d.dot(d.T) + (sum_XXtrans - xd - xd.T - term) / T.sum()

                params['R'] = (r + r.T) / 2  # ensure symmetry

            if params['notes']['learnKernelParams']:
                res = self._learn_gp_params(latent_seqs, params, verbose=verbose)
                params['gamma'] = res['gamma']

            t_end = time.time() - tic
            iter_time.append(t_end)

            # Verify that likelihood is growing monotonically
            if iter_id <= 2:
                ll_base = ll
            elif verbose and ll < ll_old:
                print('\nError: Data likelihood has decreased ',
                    'from {0} to {1}'.format(ll_old, ll))
            elif (ll - ll_base) < (1 + tol) * (ll_old - ll_base):
                break

        if len(lls) < max_iters:
            print('Fitting has converged after {0} EM iterations.)'.format(
                len(lls)))

        if np.any(np.diag(params['R']) == var_floor):
            warnings.warn('Private variance floor used for one or more observed '
                        'dimensions in GPFA.')

        return params, latent_seqs, lls, iter_time

    def _infer_latents(self, X, params, get_ll=True):
        """
        Extracts latent trajectories from observed data
        given GPFA model parameters.

        Parameters
        ----------
        X : numpy.ndarray
            input data structure, whose n-th element (corresponding to the n-th
            experimental trial) of shape (#x_dim, #bins)
        params : dict
            GPFA model parameters whe the following fields:
            C : numpy.ndarray
                FA factor loadings matrix
            d : numpy.ndarray
                FA mean vector
            R : numpy.ndarray
                FA noise covariance matrix
            gamma : numpy.ndarray
                GP timescale
            eps : numpy.ndarray
                GP noise variance
        get_ll : bool, optional
            specifies whether to compute data log likelihood (default: True)

        Returns
        -------
        latent_seqs : numpy.recarray
            X_out : numpy.ndarray
                input data structure, whose n-th element (corresponding to the n-th
                experimental trial) has fields:
                X : numpy.ndarray of shape (#x_dim, #bins)
            pZ_mu : (#z_dim, #bins) numpy.ndarray
                posterior mean of latent variables at each time bin
            pZ_cov : (#z_dim, #z_dim, #bins) numpy.ndarray
                posterior covariance between latent variables at each timepoint
            pZ_covGP : (#bins, #bins, #z_dim) numpy.ndarray
                    posterior covariance over time for each latent variable
        ll : float
            data log likelihood, numpy.nan is returned when `get_ll` is set False
        """
        x_dim, z_dim = params['C'].shape

        # copy the contents of the input data structure to output structure
        X_out = np.empty(len(X), dtype=[('X', object)])
        for s, seq in enumerate(X_out):
            seq['X'] = X[s]

        dtype_out = [(i, X_out[i].dtype) for i in X_out.dtype.names]
        dtype_out.extend([('pZ_mu', object), ('pZ_cov', object),
                        ('pZ_covGP', object)])
        latent_seqs = np.empty(len(X_out), dtype=dtype_out)
        for dtype_name in X_out.dtype.names:
            latent_seqs[dtype_name] = X_out[dtype_name]

        # Precomputations
        if params['notes']['RforceDiagonal']:
            rinv = np.diag(1.0 / np.diag(params['R']))
            logdet_r = (np.log(np.diag(params['R']))).sum()
        else:
            rinv = linalg.inv(params['R'])
            rinv = (rinv + rinv.T) / 2  # ensure symmetry
            logdet_r = fast_logdet(params['R'])

        c_rinv = params['C'].T.dot(rinv)
        c_rinv_c = c_rinv.dot(params['C'])

        t_all = [X_n.shape[1] for X_n in X]
        t_uniq = np.unique(t_all)
        ll = 0.

        # Overview:
        # - Outer loop on each element of Tu.
        # - For each element of Tu, find all trials with that length.
        # - Do inference and LL computation for all those trials together.
        for t in t_uniq:
            k_big, k_big_inv, logdet_k_big = self._make_k_big(params, t)
            k_big = sparse.csr_matrix(k_big)

            blah = [c_rinv_c for _ in range(t)]
            c_rinv_c_big = linalg.block_diag(*blah)  # (x_dim*T) x (x_dim*T)
            minv, logdet_m = self._inv_persymm(k_big_inv + c_rinv_c_big, z_dim)

            # Note that posterior covariance does not depend on observations,
            # so can compute once for all trials with same T.
            # x_dim x x_dim posterior covariance for each timepoint
            vsm = np.full((z_dim, z_dim, t), np.nan)
            idx = np.arange(0, z_dim * t + 1, z_dim)
            for i in range(t):
                vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

            # T x T posterior covariance for each GP
            vsm_gp = np.full((t, t, z_dim), np.nan)
            for i in range(z_dim):
                vsm_gp[:, :, i] = minv[i::z_dim, i::z_dim]

            # Process all trials with length T
            n_list = np.where(t_all == t)[0]
            # dif is x_dim x sum(T)
            dif = np.hstack(latent_seqs[n_list]['X']) - params['d'][:, np.newaxis]
            # term1Mat is (z_dim*T) x length(nList)
            term1_mat = c_rinv.dot(dif).reshape((z_dim * t, -1), order='F')

            # Compute blkProd = CRinvC_big * invM efficiently
            # blkProd is block persymmetric, so just compute top half
            t_half = int(np.ceil(t / 2.0))
            blk_prod = np.zeros((z_dim * t_half, z_dim * t))
            idx = range(0, z_dim * t_half + 1, z_dim)
            for i in range(t_half):
                blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c.dot(
                    minv[idx[i]:idx[i + 1], :])
            blk_prod = k_big[:z_dim * t_half, :].dot(
                self._fill_persymm(np.eye(z_dim * t_half, z_dim * t) -
                                    blk_prod, z_dim, t))
            # latent_variable Matrix (Z_mat) is (z_dim*T) x length(nList)
            Z_mat = self._fill_persymm(
                blk_prod, z_dim, t).dot(term1_mat)

            for i, n in enumerate(n_list):
                latent_seqs[n]['pZ_mu'] = \
                    Z_mat[:, i].reshape((z_dim, t), order='F')
                latent_seqs[n]['pZ_cov'] = vsm
                latent_seqs[n]['pZ_covGP'] = vsm_gp

            if get_ll:
                # Compute data likelihood
                val = -t * logdet_r - logdet_k_big - logdet_m \
                    - x_dim * t * np.log(2 * np.pi)
                ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum() \
                    + (term1_mat.T.dot(minv) * term1_mat.T).sum()

        if get_ll:
            ll /= 2
        else:
            ll = np.nan

        return latent_seqs, ll


    def _learn_gp_params(self, latent_seqs, params, verbose=False):
        """Updates parameters of GP state model, given trajectories.

        Parameters
        ----------
        latent_seqs : numpy.recarray
            data structure containing trajectories;
        params : dict
            current GP state model parameters, which gives starting point
            for gradient optimization;
        verbose : bool, optional
            specifies whether to display status messages (default: False)

        Returns
        -------
        param_opt : numpy.ndarray
            updated GP state model parameter

        Raises
        ------
        ValueError
            If `params['covType'] != 'rbf'`.
            If `params['notes']['learnGPNoise']` set to True.

        """
        if params['covType'] != 'rbf':
            raise ValueError("Only 'rbf' GP covariance type is supported.")
        if params['notes']['learnGPNoise']:
            raise ValueError("learnGPNoise is not supported.")
        param_name = 'gamma'

        param_init = params[param_name]
        param_opt = {param_name: np.empty_like(param_init)}

        z_dim = param_init.shape[-1]
        precomp = self._make_precomp(latent_seqs, z_dim)

        # Loop once for each state dimension (each GP)
        for i in range(z_dim):
            const = {'eps': params['eps'][i]}
            initp = np.log(param_init[i])
            res_opt = optimize.minimize(self._grad_betgam, initp,
                                        args=(precomp[i], const),
                                        method='L-BFGS-B', jac=True)
            param_opt['gamma'][i] = np.exp(res_opt.x)

            if verbose:
                print(f'\n Converged p; z_dim:{i}, p:{res_opt.x}')

        return param_opt


    def _orthonormalize(self, params_est, seqs):
        """
        Orthonormalize the columns of the loading matrix C and apply the
        corresponding linear transform to the latent variables.

        Parameters
        ----------
        params_est : dict
            First return value of extract_trajectory() on the training data set.
            Estimated model parameters.
            When the GPFA method is used, following parameters are contained
            covType : {'rbf', 'tri', 'logexp'}
                type of GP covariance
                Currently, only 'rbf' is supported.
            gamma : numpy.ndarray of shape (1, #z_dim)
                related to GP timescales by 'bin_size / sqrt(gamma)'
            eps : numpy.ndarray of shape (1, #z_dim)
                GP noise variances
            d : numpy.ndarray of shape (#x_dim, 1)
                observation mean
            C : numpy.ndarray of shape (#x_dim, #z_dim)
                mapping between the observational space and the latent variable
                space
            R : numpy.ndarray of shape (#x_dim, #z_dim)
                observation noise covariance

        seqs : numpy.recarray
            Contains the embedding of the training data into the latent variable
            space.
            Data structure, whose n-th entry (corresponding to the n-th
            experimental trial) has field
            X : numpy.ndarray of shape (#x_dim, #bins)
                observed data
            pZ_mu : numpy.ndarray of shape (#z_dim, #bins)
                posterior mean of latent variables at each time bin
            pZ_cov : numpy.ndarray of shape (#z_dim, #z_dim, #bins)
                posterior covariance between latent variables at each timepoint
            pZ_covGP : numpy.ndarray of shape (#bins, #bins, #z_dim)
                posterior covariance over time for each latent variable

        Returns
        -------
        params_est : dict
            Estimated model parameters, including `Corth`, obtained by
            orthonormalizing the columns of C.
        seqs : numpy.recarray
            Training data structure that contains the new field
            `pZ_mu_orth`, the orthonormalized neural trajectories.
        """
        C = params_est['C']
        Z_all = np.hstack(seqs['pZ_mu'])
        pZ_mu_orth, Corth, _ = self._orthonormalize_util(Z_all, C)
        seqs = self._segment_by_trial(seqs, pZ_mu_orth, 'pZ_mu_orth')

        params_est['Corth'] = Corth

        return Corth, seqs

    def _cut_trials(self, X_in, seg_length=20):
        """
        Extracts trial segments that are all of the same length.  Uses
        overlapping segments if trial length is not integer multiple
        of segment length.  Ignores trials with length shorter than
        one segment length.

        Parameters
        ----------
        X_in : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.

        seg_length : int
            length of segments to extract, in number of timesteps. If infinite,
            entire trials are extracted, i.e., no segmenting.
            Default: 20

        Returns
        -------
        X_out : np.ndarray
            data structure containing np.ndarrays whose n-th element
            (corresponding to the n-th segment) has shape of
            (#x_dim x #seg_length)

        Raises
        ------
        ValueError
            If `seq_length == 0`.

        """
        if seg_length == 0:
            raise ValueError("At least 1 extracted trial must be returned")
        if np.isinf(seg_length):
            X_out = X_in
            return X_out

        X_out_buff = []
        for n, X_in_n in enumerate(X_in):
            T = X_in_n.shape[1]

            # Skip trials that are shorter than segLength
            if T < seg_length:
                warnings.warn(
                    f'trial corresponding to index {n} \
                        shorter than one segLength...'
                    'skipping')
                continue

            numSeg = int(np.ceil(float(T) / seg_length))

            # Randomize the sizes of overlaps
            if numSeg == 1:
                cumOL = np.array([0, ])
            else:
                totalOL = (seg_length * numSeg) - T
                probs = np.ones(numSeg - 1, float) / (numSeg - 1)
                randOL = np.random.multinomial(totalOL, probs)
                cumOL = np.hstack([0, np.cumsum(randOL)])

            seg = np.empty(numSeg, object)

            for n_seg in range(numSeg):
                tStart = seg_length * n_seg - cumOL[n_seg]
                seg[n_seg] = X_in_n[:, tStart:tStart + seg_length]

            X_out_buff.append(seg)

        if len(X_out_buff) > 0:
            X_out = np.hstack(X_out_buff)
        else:
            X_out = np.empty(0)

        return X_out

    def _make_k_big(self, params, n_timesteps):
        """
        Constructs full GP covariance matrix across all state dimensions and
        timesteps.

        Parameters
        ----------
        params : dict
            GPFA model parameters
        n_timesteps : int
            number of timesteps

        Returns
        -------
        K_big : np.ndarray
            GP covariance matrix with dimensions (z_dim * T) x (z_dim * T).
            The (t1, t2) block is diagonal, has dimensions z_dim x z_dim, and
            represents the covariance between the state vectors at timesteps
            t1 and t2. K_big is sparse and striped.
        K_big_inv : np.ndarray
            Inverse of K_big
        logdet_K_big : float
            Log determinant of K_big

        Raises
        ------
        ValueError
            If `params['covType'] != 'rbf'`.

        """
        if params['covType'] != 'rbf':
            raise ValueError("Only 'rbf' GP covariance type is supported.")

        z_dim = params['C'].shape[1]

        K_big = np.zeros((z_dim * n_timesteps, z_dim * n_timesteps))
        K_big_inv = np.zeros((z_dim * n_timesteps, z_dim * n_timesteps))
        Tdif = np.tile(np.arange(0, n_timesteps), (n_timesteps, 1)).T \
            - np.tile(np.arange(0, n_timesteps), (n_timesteps, 1))
        logdet_K_big = 0

        for i in range(z_dim):
            K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 *
                                                Tdif ** 2) \
                + params['eps'][i] * np.eye(n_timesteps)
            K_big[i::z_dim, i::z_dim] = K
            K_big_inv[i::z_dim, i::z_dim] = np.linalg.inv(K)
            logdet_K = fast_logdet(K)

            logdet_K_big = logdet_K_big + logdet_K

        return K_big, K_big_inv, logdet_K_big


    def _inv_persymm(self, M, blk_size):
        """
        Inverts a matrix that is block persymmetric.  This function is
        faster than calling inv(M) directly because it only computes the
        top half of inv(M).  The bottom half of inv(M) is made up of
        elements from the top half of inv(M).

        WARNING: If the input matrix M is not block persymmetric, no
        error message will be produced and the output of this function will
        not be meaningful.

        Parameters
        ----------
        M : (blkSize*T, blkSize*T) np.ndarray
            The block persymmetric matrix to be inverted.
            Each block is blkSize x blkSize, arranged in a T x T grid.
        blk_size : int
            Edge length of one block

        Returns
        -------
        invM : (blkSize*T, blkSize*T) np.ndarray
            Inverse of M
        logdet_M : float
            Log determinant of M
        """
        T = int(M.shape[0] / blk_size)
        Thalf = int(np.ceil(T / 2.0))
        mkr = blk_size * Thalf

        invA11 = np.linalg.inv(M[:mkr, :mkr])
        invA11 = (invA11 + invA11.T) / 2

        # Multiplication of a sparse matrix by a dense matrix is not supported by
        # SciPy. Making A12 a sparse matrix here  an error later.
        off_diag_sparse = False
        if off_diag_sparse:
            A12 = sp.sparse.csr_matrix(M[:mkr, mkr:])
        else:
            A12 = M[:mkr, mkr:]

        term = invA11.dot(A12)
        F22 = M[mkr:, mkr:] - A12.T.dot(term)

        res12 = np.linalg.solve(F22.T, -term.T).T
        res11 = invA11 - res12.dot(term.T)
        res11 = (res11 + res11.T) / 2

        # Fill in bottom half of invM by picking elements from res11 and res12
        invM = self._fill_persymm(np.hstack([res11, res12]), blk_size, T)

        logdet_M = -fast_logdet(invA11) + fast_logdet(F22)

        return invM, logdet_M


    def _fill_persymm(self, p_in, blk_size, n_blocks, blk_size_vert=None):
        """
        Fills in the bottom half of a block persymmetric matrix, given the
        top half.

        Parameters
        ----------
        p_in :  (x_dim*Thalf, x_dim*T) np.ndarray
            Top half of block persymmetric matrix, where Thalf = ceil(T/2)
        blk_size : int
            Edge length of one block
        n_blocks : int
            Number of blocks making up a row of Pin
        blk_size_vert : int, optional
            Vertical block edge length if blocks are not square.
            `blk_size` is assumed to be the horizontal block edge length.

        Returns
        -------
        Pout : (z_dim*T, z_dim*T) np.ndarray
            Full block persymmetric matrix
        """
        if blk_size_vert is None:
            blk_size_vert = blk_size

        Nh = blk_size * n_blocks
        Nv = blk_size_vert * n_blocks
        Thalf = int(np.floor(n_blocks / 2.0))
        THalf = int(np.ceil(n_blocks / 2.0))

        Pout = np.empty((blk_size_vert * n_blocks, blk_size * n_blocks))
        Pout[:blk_size_vert * THalf, :] = p_in
        for i in range(Thalf):
            for j in range(n_blocks):
                Pout[Nv - (i + 1) * blk_size_vert:Nv - i * blk_size_vert,
                    Nh - (j + 1) * blk_size:Nh - j * blk_size] \
                    = p_in[i * blk_size_vert:(i + 1) *
                        blk_size_vert,
                        j * blk_size:(j + 1) * blk_size]

        return Pout


    def _make_precomp(self, Seqs, z_dim):
        """
        Make the precomputation matrices specified by the GPFA algorithm.

        Parameters
        ----------
        Seqs : numpy.recarray
            The sequence struct of inferred latents, etc.
        z_dim : int
        The dimension of the latent space.

        Returns
        -------
        precomp : numpy.recarray
            The precomp struct will be updated with the posterior covaraince and
            the other requirements.

        Notes
        -----
        All inputs are named sensibly to those in `learnGPparams`.
        This code probably should not be called from anywhere but there.

        We bother with this method because we
        need this particular matrix sum to be
        as fast as possible.  Thus, no error checking
        is done here as that would add needless computation.
        Instead, the onus is on the caller (which should be
        learnGPparams()) to make sure this is called correctly.

        Finally, see the notes in the GPFA README.
        """

        Tall = np.array([X_n.shape[1] for X_n in Seqs['X']])
        Tmax = max(Tall)
        Tdif = np.tile(np.arange(0, Tmax), (Tmax, 1)).T \
            - np.tile(np.arange(0, Tmax), (Tmax, 1))

        # assign some helpful precomp items
        # this is computationally cheap, so we keep a few loops in MATLAB
        # for ease of readability.
        precomp = np.empty(z_dim, dtype=[(
            'absDif', object), ('difSq', object), ('Tmax', object),
            ('Tu', object)])
        for i in range(z_dim):
            precomp[i]['absDif'] = np.abs(Tdif)
            precomp[i]['difSq'] = Tdif ** 2
            precomp[i]['Tmax'] = Tmax
        # find unique numbers of trial lengths
        trial_lengths_num_unique = np.unique(Tall)
        # Loop once for each state dimension (each GP)
        for i in range(z_dim):
            precomp_Tu = np.empty(len(trial_lengths_num_unique), dtype=[(
                'nList', object), ('T', int), ('numTrials', int),
                ('PautoSUM', object)])
            for j, trial_len_num in enumerate(trial_lengths_num_unique):
                precomp_Tu[j]['nList'] = np.where(Tall == trial_len_num)[0]
                precomp_Tu[j]['T'] = trial_len_num
                precomp_Tu[j]['numTrials'] = len(precomp_Tu[j]['nList'])
                precomp_Tu[j]['PautoSUM'] = np.zeros((trial_len_num,
                                                    trial_len_num))
                precomp[i]['Tu'] = precomp_Tu

        ############################################################
        # Fill out PautoSum
        ############################################################
        # Loop once for each state dimension (each GP)
        for i in range(z_dim):
            # Loop once for each trial length (each of Tu)
            for j in range(len(trial_lengths_num_unique)):
                # Loop once for each trial (each of nList)
                for n in precomp[i]['Tu'][j]['nList']:
                    precomp[i]['Tu'][j]['PautoSUM'] += \
                        Seqs[n]['pZ_covGP'][:, :, i] \
                        + np.outer(Seqs[n]['pZ_mu'][i, :], Seqs[n]['pZ_mu'][i, :])
        return precomp


    def _grad_betgam(self, p, pre_comp, const):
        """
        Gradient computation for GP timescale optimization.
        This function is called by minimize.m.

        Parameters
        ----------
        p : float
            variable with respect to which optimization is performed,
            where :math:`p = log(1 / timescale^2)`
        pre_comp : numpy.recarray
            structure containing precomputations
        const : dict
            contains hyperparameters

        Returns
        -------
        f : float
            value of objective function E[log P({x},{y})] at p
        df : float
            gradient at p
        """
        Tmax = pre_comp['Tmax']

        # temp is Tmax x Tmax
        temp = (1 - const['eps']) * np.exp(-np.exp(p) / 2 * pre_comp['difSq'])
        Kmax = temp + const['eps'] * np.eye(Tmax)
        dKdgamma_max = -0.5 * temp * pre_comp['difSq']

        dEdgamma = 0
        f = 0
        for j in range(len(pre_comp['Tu'])):
            T = pre_comp['Tu'][j]['T']
            Thalf = int(np.ceil(T / 2.0))

            Kinv = np.linalg.inv(Kmax[:T, :T])
            logdet_K = fast_logdet(Kmax[:T, :T])

            KinvM = Kinv[:Thalf, :].dot(dKdgamma_max[:T, :T])  # Thalf x T
            KinvMKinv = (KinvM.dot(Kinv)).T  # Thalf x T

            dg_KinvM = np.diag(KinvM)
            tr_KinvM = 2 * dg_KinvM.sum() - np.fmod(T, 2) * dg_KinvM[-1]

            mkr = int(np.ceil(0.5 * T ** 2))
            numTrials = pre_comp['Tu'][j]['numTrials']
            PautoSUM = pre_comp['Tu'][j]['PautoSUM']

            pauto_kinv_dot = PautoSUM.ravel('F')[:mkr].dot(
                KinvMKinv.ravel('F')[:mkr])
            pauto_kinv_dot_rest = PautoSUM.ravel('F')[-1:mkr - 1:- 1].dot(
                KinvMKinv.ravel('F')[:(T ** 2 - mkr)])
            dEdgamma = dEdgamma - 0.5 * numTrials * tr_KinvM \
                + 0.5 * pauto_kinv_dot \
                + 0.5 * pauto_kinv_dot_rest

            f = f - 0.5 * numTrials * logdet_K \
                - 0.5 * (PautoSUM * Kinv).sum()

        f = -f
        # exp(p) is needed because we're computing gradients with
        # respect to log(gamma), rather than gamma
        df = -dEdgamma * np.exp(p)

        return f, df


    def _orthonormalize_util(self, Z, l_mat):
        """
        Orthonormalize the columns of the loading matrix and apply the
        corresponding linear transform to the latent variables.
        In the following description, z_dim and x_dim refer to data dimensionality
        and latent dimensionality, respectively.

        Parameters
        ----------
        Z :  (z_dim, T) numpy.ndarray
            Latent variables
        l_mat :  (x_dim, z_dim) numpy.ndarray
            Loading matrix

        Returns
        -------
        pZ_mu_orth : (x_dim, T) numpy.ndarray
            Orthonormalized latent variables
        Lorth : (z_dim, x_dim) numpy.ndarray
            Orthonormalized loading matrix
        TT :  (x_dim, x_dim) numpy.ndarray
        Linear transform applied to latent variables
        """
        z_dim = l_mat.shape[1]
        if z_dim == 1:
            TT = np.sqrt(np.dot(l_mat.T, l_mat))
            Lorth = np.linalg.solve(TT.T, l_mat.T).T
            pZ_mu_orth = np.dot(TT, Z)
        else:
            UU, DD, VV = sp.linalg.svd(l_mat, full_matrices=False)
            # TT is transform matrix
            TT = np.dot(np.diag(DD), VV)

            Lorth = UU
            pZ_mu_orth = np.dot(TT, Z)
        return pZ_mu_orth, Lorth, TT


    def _segment_by_trial(self, seqs, Z, fn):
        """
        Segment and store data by trial.

        Parameters
        ----------
        seqs : numpy.recarray
            Data structure that has field Z, the observations
        Z : numpy.ndarray
            Data to be segmented (any dimensionality Z total number of timesteps)
        fn : str
            New field name of seq where segments of X are stored

        Returns
        -------
        seqs_new : numpy.recarray
            Data structure with new field `fn`

        Raises
        ------
        ValueError
            If "`All timespets` != Z.shape[1]".

        """
        T_all = [X_n.shape[1] for X_n in seqs['X']]
        if np.sum(T_all) != Z.shape[1]:
            raise ValueError('size of X incorrect.')

        dtype_new = [(i, seqs[i].dtype) for i in seqs.dtype.names]
        dtype_new.append((fn, object))
        seqs_new = np.empty(len(seqs), dtype=dtype_new)
        for dtype_name in seqs.dtype.names:
            seqs_new[dtype_name] = seqs[dtype_name]

        ctr = 0
        for n, T_n in enumerate(T_all):
            seqs_new[n][fn] = Z[:, ctr:ctr + T_n]
            ctr += T_n

        return seqs_new    
