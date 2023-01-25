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

2) projection of single trials in the low dimensional space and perform
orthonormalization of the matrix C

3) prediction and orthonormalization of the corresponding subspace,
for visualization purposes

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
import sklearn
import scipy as sp
import numpy as np
from tqdm import trange
from sklearn.base import clone
import scipy.linalg as linalg
import scipy.optimize as optimize
from sklearn.utils.extmath import fast_logdet
from sklearn.decomposition import FactorAnalysis
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel

__all__ = [
    "GPFA"
]


class GPFA(sklearn.base.BaseEstimator):
    """
    Apply Gaussian process factor analysis (GPFA) to observed data

    There are two principle scenarios of using the GPFA analysis, both of which
    can be performed in an instance of the GPFA() class.

    In the first scenario, only one single dataset is used to fit the model and
    to extract the trajectories. The parameters that describe the
    transformation are first extracted from the data using the `fit()` method
    of the GPFA class. Then the same data is projected into the orthonormal
    basis using the method `predict()`.

    In the second scenario, a single dataset is split into training and test
    datasets. Here, the parameters are estimated from the training data. Then
    the test data is projected into the low-dimensional space previously
    obtained from the training data. This analysis is performed by executing
    first the `fit()` method on the training data, followed by the
    `predict()` method on the test dataset.

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
    gp_kernel : kernel instance, default=None
        If None is passed, the kernel defaults to
        ConstantKernel(1-0.001, constant_value_bounds='fixed')
        * RBF(length_scale=0.1)
        + ConstantKernel(0.001, constant_value_bounds='fixed')
        * WhiteKernel(noise_level=1, noise_level_bounds='fixed')
        where only kernel hyperparameters not marked as 'fixed'
        are learned - in this case only the `length scale`
        of the RBF kernel.
        Note that the `gp_kernel` can either be a single kernel
        (in which case it will be replicated across all latent dimensions),
        or a sequence of kernels, one per latent dimension
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
    freq_ll : int, optional
        data likelihood is computed at every freq_ll EM iterations. freq_ll = 1
        means that data likelihood is computed at every iteration.
        Default: 5
    verbose : bool, optional
        specifies whether to display status messages
        Default: False

    Attributes
    ----------
    valid_data_names_ : tuple of str
        Names of the data contained in the resultant data structure, used to
        check the validity of users' request
    Estimated model parameters. Updated when calling `fit()` method.
        self.gp_kernel.theta : numpy.array
            the flattened and log-transformed non-fixed hyperparams
            to which optimization is performed,
            where :math:`theta = log(kernel parameters)`
        d_ : (#x_dim, 1) numpy.ndarray
            observation mean
        C_ : (#x_dim, #z_dim) numpy.ndarray
            loading matrix, representing the mapping between the observed data
            space and the latent variable space
        R_ : (#x_dim, #x_dim) numpy.ndarray
            observation noise covariance
    fit_info_ : dict
        Information of the fitting process. Updated at each run of the fit()
        method.
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
        log_likelihoods : list
            log likelihoods after each EM iteration.
    train_latent_seqs_ : numpy.recarray
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

    Methods
    -------
    fit
    predict
    score

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
    >>> results = gpfa.predict(X,
    ...                        returned_data=['pZ_mu_orth', 'pZ_mu'])
    >>> pZ_mu_orth = results['pZ_mu_orth']
    >>> pZ_mu = results['pZ_mu']

    """

    def __init__(self, bin_size=0.02, gp_kernel=None, z_dim=3,
                 min_var_frac=0.01, em_tol=1.0E-8, em_max_iters=500,
                 freq_ll=5, verbose=False):
        self.bin_size = bin_size
        self.z_dim = z_dim
        self.gp_kernel = gp_kernel
        self.min_var_frac = min_var_frac
        self.em_tol = em_tol
        self.em_max_iters = em_max_iters
        self.freq_ll = freq_ll
        self.valid_data_names_ = (
            'pZ_mu_orth',
            'pZ_mu',
            'pZ_cov',
            'pZ_covGP',
            'X')
        self.verbose = verbose

        # ==================================
        # Initialize state model parameters
        # ==================================
        # will be updated later
        if self.gp_kernel is None:  # Use an RBF kernel as default
            self.gp_kernel = ConstantKernel(
                1-0.001, constant_value_bounds='fixed'
                ) * RBF(length_scale=0.1) + ConstantKernel(
                    0.001, constant_value_bounds='fixed'
                    ) * WhiteKernel(
                        noise_level=1, noise_level_bounds='fixed'
                            )

        if isinstance(self.gp_kernel, Kernel):
            self.gp_kernel = [
                clone(self.gp_kernel) for _ in range(self.z_dim)
                ]
        elif len(self.gp_kernel) != self.z_dim:
            raise ValueError(
                "The sequence length of gp_kernel: "
                f"{len(self.gp_kernel)}, doesn't match with the "
                f"number of latent dimensions: {self.z_dim}."
                )
        self.fit_info_ = {}
        self.train_latent_seqs_ = None

    def fit(self, X, use_cut_trials=False):
        """
        Fit the model with the given training data. This ceates and adjusts
        all the attributes by EM algorithm. And applies an orthonormalization
        transform to the loading matrxi.

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
        # ====================================================================
        # Cut trials: Extracts trial segments that are all of the same length.
        # ====================================================================
        X_in = X
        if use_cut_trials:
            # For compute efficiency, train on shorter segments of trials
            X_in = self._cut_trials(X_in)
            if len(X_in) == 0:
                warnings.warn('No segments extracted for training. Defaulting '
                              'to segLength=Inf.')
                X_in = self._cut_trials(X_in, seg_length=np.inf)

        # =================================================
        # Check if training data's covariance is full rank.
        # =================================================
        X_all = np.hstack(X_in)
        x_dim = X_all.shape[0]

        if np.linalg.matrix_rank(np.cov(X_all)) < x_dim:
            errmesg = 'Observation covariance matrix is rank deficient.\n' \
                      'Possible causes: ' \
                      'repeated units, not enough observations.'
            raise ValueError(errmesg)

        if self.verbose:
            print(f'Number of training trials: {len(X_in)}')
            print(f'Latent space dimensionality: {self.z_dim}')
            print(f'Observation dimensionality: {X_all.any(axis=1).sum()}')

        # ========================================
        # Initialize observation model parameters
        # ========================================
        print('Initializing parameters using factor analysis...')

        fa = FactorAnalysis(
                        n_components=self.z_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(X_all, bias=True))
                        )
        fa.fit(X_all.T)
        self.d_ = X_all.mean(axis=1)
        self.C_ = fa.components_.T
        self.R_ = np.diag(fa.noise_variance_)

        # Define parameter constraints
        self.notes_ = {
            'learnKernelParams': True,
            'RforceDiagonal': True,
        }

        # =====================
        # Fit model parameters
        # =====================
        print('\nFitting GPFA model...')

        self.train_latent_seqs_ = self._em(X_in)
        # If `use_cut_trials=True` re-compute the latent sequence on a full
        # X rather than on the cut_trial
        if use_cut_trials:
            self.train_latent_seqs_, _ = self._infer_latents(X)

        # ===========================================
        # compute the orthonormalization parameters.
        # ===========================================

        # Orthonormalize the columns of the loading matrix `C`.
        if self.z_dim == 1:
            # OrthTrans_ is transform matrix            
            self.OrthTrans_ = np.sqrt(np.dot(self.C_.T, self.C_))
            # Orthonormalized loading matrix
            Corth = np.linalg.solve(self.OrthTrans_.T, self.C_.T).T

        else:
            UU, DD, VV = sp.linalg.svd(self.C_, full_matrices=False)
            self.OrthTrans_ = np.dot(np.diag(DD), VV)
            # Orthonormalized loading matrix
            Corth = UU

        self.Corth_ = Corth

        return self

    def predict(self, X=None, returned_data=['pZ_mu_orth']):
        """
        Obtain trajectories of in a low-dimensional latent variable space by
        inferring the posterior mean of the obtained GPFA model and applying
        an orthonormalization on the latent variable space.

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None

            Note: If X=None, the latent state estimates for the training
                set are returned.

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

        lls : float
            data log likelihoods

        Raises
        ------
        ValueError
            If `returned_data` contains keys different from the ones in
            `self.valid_data_names_`.
        """
        invalid_keys = set(returned_data).difference(self.valid_data_names_)
        if len(invalid_keys) > 0:
            raise ValueError("'returned_data' can only have the following "
                             f"entries: {self.valid_data_names_}")
        if X is None:
            seqs = self.train_latent_seqs_
            lls = self.fit_info_['log_likelihoods']
        else:
            seqs, lls = self._infer_latents(X, get_ll=True)
        if 'pZ_mu_orth' in returned_data:
            seqs = self._orthonormalize(seqs)
        if len(returned_data) == 1:
            return seqs[returned_data[0]], lls
        return {i: seqs[i] for i in returned_data}, lls

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
            Log-likelihood of the given X under the fitted model.
        """
        _, ll = self.predict(X, returned_data=['pZ_mu'])
        return ll

    def _em(self, X):
        """
        Fits GPFA model parameter attributes using expectation-maximization
        (EM) algorithm.
        And also updates `self.fit_info_`

        Parameters
        ----------
        X   : an array-like of observation sequences, one per trial.
            Each element in X is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X, but #bins
            can be different for each observation sequence.
            Default : None

        Returns
        -------
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
        """
        lls = []
        ll_old = ll_base = ll = 0.0
        iter_time = []
        var_floor = self.min_var_frac * np.diag(np.cov(np.hstack(X)))
        Tall = np.array([X_n.shape[1] for X_n in X])
        Tmax = max(Tall)
        Tsdt = np.arange(0, Tmax) * self.bin_size
        unique_Ts = np.unique(Tall)

        # ============== Make Precomp_init ==============
        # assign some helpful precomp items
        precomp = {'Tsdt': Tsdt[:,np.newaxis], 'Tu': np.empty(len(unique_Ts),
                dtype=[('nList', object), ('T', int), ('numTrials', int),
                ('PautoSUM', object)])}

        # Loop once for each unique trial length
        for j, trial_len_num in enumerate(unique_Ts):
            precomp['Tu'][j]['nList'] = np.where(Tall == trial_len_num)[0]
            precomp['Tu'][j]['T'] = trial_len_num
            precomp['Tu'][j]['numTrials'] = len(precomp['Tu'][j]['nList'])
            precomp['Tu'][j]['PautoSUM'] = np.empty(
                (self.z_dim, precomp['Tu'][j]['T'], precomp['Tu'][j]['T']))

        # Loop once for each iteration of EM algorithm
        for iter_id in trange(1, self.em_max_iters + 1, desc='EM iteration',
                              disable=not self.verbose):

            if self.verbose:
                print()
            tic = time.time()
            get_ll = (np.fmod(iter_id, self.freq_ll) == 0) or (iter_id <= 2)

            # ==== E STEP =====
            if not np.isnan(ll):
                ll_old = ll
            if get_ll:
                latent_seqs, ll = self._infer_latents(X)
            else:
                latent_seqs = self._infer_latents(X, get_ll=False)
                ll = np.nan
            lls.append(ll)

            # ==== M STEP ====
            sum_p_auto = np.zeros((self.z_dim, self.z_dim))
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
            term = np.vstack(
                    [np.hstack([sum_p_auto, sum_Zall]),
                    np.hstack([sum_Zall.T, Tall.sum().reshape((1, 1))])]
                    )
            # x_dim x (z_dim+1)
            cd = np.linalg.solve(
                term.T, np.hstack([sum_XZtrans, sum_Xall]).T).T

            self.C_ = cd[:, :self.z_dim]
            self.d_ = cd[:, -1]

            # xd must be based on the new d
            # xd = X * d
            # R = (X * X.T - 2 * xd * (
            #   (sum_XZtrans - d.dot(sum_Zall.T)) * c).sum(axis=1)
            #   ) * Tall.sum()
            c = self.C_
            d = self.d_[:, np.newaxis]
            if self.notes_['RforceDiagonal']:
                sum_XXtrans = (X_all * X_all).sum(axis=1)[:, np.newaxis]
                xd = sum_Xall * d
                term = ((sum_XZtrans - d.dot(sum_Zall.T)) * c).sum(axis=1)
                term = term[:, np.newaxis]
                r = d ** 2 + (sum_XXtrans - 2 * xd - term) / Tall.sum()

                # Set minimum private variance
                r = np.maximum(var_floor, r)
                self.R_ = np.diag(r[:, 0])
            else:
                sum_XXtrans = X_all.dot(X_all.T)
                xd = sum_Xall.dot(d.T)
                term = (sum_XZtrans - d.dot(sum_Zall.T)).dot(c.T)
                r = d.dot(d.T) + (sum_XXtrans - xd - xd.T - term) / Tall.sum()

                self.R_ = (r + r.T) / 2  # ensure symmetry

            if self.notes_['learnKernelParams']:
                self._learn_gp_params(latent_seqs, precomp)

            t_end = time.time() - tic
            iter_time.append(t_end)

            # Verify that likelihood is growing monotonically
            if iter_id <= 2:
                ll_base = ll
            elif self.verbose and ll < ll_old:
                print('\nError: Data likelihood has decreased ',
                    f'from {ll_old} to {ll}')
            elif (ll - ll_base) < (1 + self.em_tol) * (ll_old - ll_base):
                break

        if len(lls) < self.em_max_iters:
            print(f'Fitting has converged after {len(lls)} EM iterations.')

        if np.any(np.diag(self.R_) == var_floor):
            warnings.warn('Private variance floor used for one or more observed '
                        'dimensions in GPFA.')

        self.fit_info_ = {'iteration_time': iter_time, 'log_likelihoods': lls}

        return latent_seqs

    def _infer_latents(self, X, get_ll=True):
        """
        Extracts latent trajectories from observed data
        given GPFA model parameters.

        Parameters
        ----------
        X : numpy.ndarray
            input data structure, whose n-th element (corresponding to the n-th
            experimental trial) of shape (#x_dim, #bins)
        get_ll : bool, optional
            specifies whether to compute data log likelihood (default: True)
            Default : True
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
            data log likelihood, returned when `get_ll` is set True
        """
        x_dim = self.C_.shape[0]

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
        if self.notes_['RforceDiagonal']:
            rinv = np.diag(1.0 / np.diag(self.R_))
            logdet_r = (np.log(np.diag(self.R_))).sum()
        else:
            rinv = linalg.inv(self.R_)
            rinv = (rinv + rinv.T) / 2  # ensure symmetry
            logdet_r = fast_logdet(self.R_)

        c_rinv = self.C_.T.dot(rinv)
        c_rinv_c = c_rinv.dot(self.C_)

        # Get all trial lengths and find the unique lengths
        # Find the maximum trial length
        Tall = [X_n.shape[1] for X_n in X]
        unique_Ts = np.unique(Tall)
        Tmax = max(unique_Ts)
        ll = 0.

        K_big = self._make_k_big(Tmax)
        blah = [c_rinv_c for _ in range(Tmax)]
        C_rinv_c_big = linalg.block_diag(*blah)  # (z_dim*T) x (z_dim*T)

        # Overview:
        # - Outer loop on each element of Tu.
        # - Do inference and LL computation for all those trials together.
        for t in unique_Ts:
            if t == unique_Ts[0]:
                K_big_inv = linalg.inv(K_big[:t * self.z_dim, :t * self.z_dim])
                logdet_k_big = fast_logdet(K_big[:t * self.z_dim, :t * self.z_dim])
                M = K_big_inv + C_rinv_c_big[:t * self.z_dim,:t * self.z_dim]
                M_inv = linalg.inv(M)
                logdet_M = fast_logdet(M)
            else:
                # Here, we compute the inverse of K for the current t from its known
                # inverse for the previous t, using block matrix inversion identities.
                # We also use those to update the previously computed M_inv by using
                # the (top-left block of the) new K_big_inv rather than the K_big_inv
                # for the previous t. This updated M_inv (here called MAinv) is in
                # turn used to compute the M_inv for the current t using similar block
                # matrix inversion identities.
                K_big_inv, logdet_k_big, MAinv, logdet_MAinv = self._sym_block_inversion(
                    K_big[:t * self.z_dim, :t * self.z_dim], K_big_inv, -logdet_k_big,
                    M_inv, -logdet_M
                    )             
                M = K_big_inv + C_rinv_c_big[:t * self.z_dim,:t * self.z_dim]
                M_inv, logdet_M = self._sym_block_inversion(M, MAinv, logdet_MAinv)

            # Note that posterior covariance does not depend on observations,
            # so can compute once for all trials with same T.
            # z_dim x z_dim x T posterior covariance for each timepoint
            vsm = np.full((self.z_dim, self.z_dim, t), np.nan)
            idx = np.arange(0, self.z_dim * t + 1, self.z_dim)
            for i in range(t):
                vsm[:, :, i] = M_inv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

            # T x T posterior covariance for each GP
            vsm_gp = np.full((self.z_dim, t, t), np.nan)
            for i in range(self.z_dim):
                vsm_gp[i, :, :] = M_inv[i::self.z_dim, i::self.z_dim]

            # Process all trials with length T
            n_list = np.where(Tall == t)[0]
            # dif is x_dim x sum(T)
            dif = np.hstack(latent_seqs[n_list]['X']) - self.d_[:, np.newaxis]
            # term1Mat is (z_dim*T) x length(nList)
            term1_mat = c_rinv.dot(dif).reshape((self.z_dim * t, -1), order='F')

            # latent_variable Matrix (Z_mat) is (z_dim*T) x length(nList)
            Z_mat = M_inv.dot(term1_mat)

            for i, n in enumerate(n_list):
                latent_seqs[n]['pZ_mu'] = \
                    Z_mat[:, i].reshape((self.z_dim, t), order='F')
                latent_seqs[n]['pZ_cov'] = vsm
                latent_seqs[n]['pZ_covGP'] = vsm_gp

            if get_ll:
                # Compute data likelihood
                val = -t * logdet_r - logdet_k_big - logdet_M \
                    - x_dim * t * np.log(2 * np.pi)
                ll = ll + len(n_list) * val - (rinv.dot(dif) * dif).sum() \
                    + (term1_mat.T.dot(M_inv) * term1_mat.T).sum()

        if get_ll:
            ll /= 2
            return latent_seqs, ll

        return latent_seqs

    def _learn_gp_params(self, latent_seqs, precomp):
        """Updates parameters of GP state model, given trajectories.

        Parameters
        ----------
        latent_seqs : numpy.recarray
            data structure containing trajectories
        precomp : numpy.recarray
            The precomp structure will be updated with the
            posterior covariance
        """
        precomp = self._fill_p_auto_sum(latent_seqs, precomp)

        # Loop once for each state dimension (each GP)
        for i in range(self.z_dim):
            gp_kernel_i = self.gp_kernel[i]
            init_theta = self.gp_kernel[i].theta
            res_opt = optimize.minimize(
                        self._grad_bet_theta,
                        init_theta,
                        args=(gp_kernel_i, precomp, i),
                        method='L-BFGS-B',
                        jac=True
                        )
            self.gp_kernel[i].theta = res_opt.x

            for j in range(len(precomp['Tu'])):
                precomp['Tu'][j]['PautoSUM'][i, :, :].fill(0)
            if self.verbose:
                print(f'\n Converged theta; z_dim:{i}, theta:{res_opt.x}')

    def _orthonormalize(self, seqs):
        """
        Apply the corresponding linear transform to the latent variables.

        Parameters
        ----------
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
        seqs : numpy.recarray
            Training data structure that contains the new field
            `pZ_mu_orth`, the orthonormalized trajectories.
        """
        Z_all = np.hstack(seqs['pZ_mu'])
        pZ_mu_orth = np.dot(self.OrthTrans_, Z_all)
        # add the field `pZ_mu_orth` to seq
        seqs = self._segment_by_trial(seqs, pZ_mu_orth, 'pZ_mu_orth')

        return seqs

    def _cut_trials(self, X_in, seg_length=20):
        """
        Extracts trial segments that are all of the same length.  Uses
        overlapping segments if trial length is not integer multiple
        of segment length.  Ignores trials with length shorter than
        one segment length.

        Parameters
        ----------
        X_in : an array-like of observation sequences, one per trial.
            Each element in X_in is a matrix of size #x_dim x #bins,
            containing an observation sequence. The input dimensionality
            #x_dim needs to be the same across elements in X_in, but #bins
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

    def _sym_block_inversion(self, M, Ainv, logdet_Ainv, X=None, logdet_X=None):
        """
        Inverts the symmetric matrix M,
                      [ A   B ]^-1   [ MAinv   MBinv ]
        Minv = M^-1 = [       ]    = [               ]
                      [ B^T D ]      [ MBinv^T MDinv ]
        exploiting the existing knowledge of Ainv = A^-1 and its
        log-determinant logdet_Ainv = log(|A^-1|) to speed up the computation.
        This function is faster than calling inv(M) directly. It also supports
        a faster computation of (MAinv + X)^-1 for some X, if given, where X
        must be the same size as Ainv.

        Parameters
        ----------
        M : numpy.ndarray
            The symmetric matrix to be inverted.
        Ainv : numpy.ndarray
            The (symmetric) inverse of the top-left block of M.
        logdet_Ainv : float
            The log-determinant of A^-1 already known.
        X : numpy.ndarray, optional
            An arbitrary matrix of the same size as Ainv.
        logdet_X : float, optional
            The log-determinant of X already known.
        Returns
        -------
        Minv : numpy.ndarray
            Inverse of M
        logdet_M : float
            Log-determinant of M
        MAinvPXinv : numpy.ndarray
            The inverse of (MAinv + X)
        loget_MAinv : float
            Log-determinant of MAinvPXinv
        """
        t = len(Ainv)
        B = M[:t, t:]
        D = M[t:, t:]
        AinvB = Ainv @ B
        MD = D - AinvB.T @ B
        MDinv = linalg.inv(MD)
        MCinv = - MDinv @ AinvB.T
        MAinv = Ainv + AinvB @ - MCinv

        Minv = np.block([
            [MAinv, MCinv.T],
            [MCinv, MDinv]
        ])

        logdet_MD = fast_logdet(MD)
        logdet_M = -logdet_Ainv + logdet_MD
        if X is not None:
            if logdet_X is None:
                logdet_X = fast_logdet(X)
            MDpAinvBXAinvB = MD + AinvB.T @ X @ AinvB
            MAinv = X - X @ AinvB @ linalg.inv(MDpAinvBXAinvB) @ AinvB.T @ X
            logdet_MAinv = logdet_MD + logdet_X - fast_logdet(MDpAinvBXAinvB)
            return Minv, logdet_M, MAinv, logdet_MAinv
        return Minv, logdet_M

    def _make_k_big(self, n_timesteps):
        """
        Constructs full GP covariance matrix across all state dimensions and
        timesteps.
        Parameters
        ----------
        n_timesteps : int
            number of timesteps
        Returns
        -------
        K_big : numpy.ndarray
            GP covariance matrix with dimensions (z_dim * T) x (z_dim * T).
            The (t1, t2) block is diagonal, has dimensions z_dim x z_dim, and
            represents the covariance between the state vectors at timesteps
            t1 and t2. K_big is sparse and striped.
        """

        K_big = np.zeros((self.z_dim * n_timesteps, self.z_dim * n_timesteps))
        tsdt = np.arange(0, n_timesteps) * self.bin_size

        for i in range(self.z_dim):
            K = self.gp_kernel[i](tsdt[:,np.newaxis])
            K_big[i::self.z_dim, i::self.z_dim] = K

        return K_big

    def _fill_p_auto_sum(self, Seqs, precomp):
        """
        Fill the PautoSUM item in precomp

        Parameters
        ----------
        Seqs : numpy.recarray
            The sequence structure of inferred latents, etc.
        precomp : numpy.recarray
            The precomp structure will be updated with the
            posterior covariance.

        Returns
        -------
        precomp : numpy.recarray
            Structure containing precomputations.

        Notes
        -----
        All inputs are named sensibly to those in `_learn_gp_params`.
        This code probably should not be called from anywhere but there.

        We bother with this method because we need this particular matrix sum to
        be as fast as possible.  Thus, no error checking is done here as that
        would add needless computation. Instead, the onus is on the caller (which
        should be `_learn_gp_params()`) to make sure this is called correctly.

        Finally, see the notes in the GPFA README.
        """
        ############################################################
        # Fill out PautoSum
        ############################################################
        # Loop once for each state dimension (each GP)
        for i in range(self.z_dim):
            # Loop once for each trial length (each of Tu)
            for j in range(len(precomp['Tu'])):
                # Loop once for each trial (each of nList)
                for n in precomp['Tu'][j]['nList']:
                    precomp['Tu'][j]['PautoSUM'][i, :, :] += \
                        Seqs[n]['pZ_covGP'][i, :, :] \
                        + np.outer(
                            Seqs[n]['pZ_mu'][i, :], Seqs[n]['pZ_mu'][i, :]
                            )
        return precomp

    def _grad_bet_theta(self, theta, gp_kernel_i, precomp, i):
        """
        Gradient computation for GP timescale optimization.
        This function is called by `_learn_gp_params()`

        Parameters
        ----------
        theta : numpy.array
            the flattened and log-transformed non-fixed hyperparams
            to which optimization is performed,
            where :math:`theta = log(kernel parameters)`
        gp_kernel_i : kernel instance
            the i-th GP kernel corresponding to the i-th latent variable
        precomp : numpy.recarray
            structure containing precomputations
        ith_zdim_index : int
            The i-th index of the i-th latent variable

        Returns
        -------
        f : numpy.array
            values of objective function E[log P({x},{y})] at theta
        df : numpy.array
            gradients at theta
        """
        gp_kernel_i.theta = theta
        Kmax, K_gradient = gp_kernel_i(
            precomp['Tsdt'], eval_gradient=True
            )
        dEdtheta, f = np.zeros(len(theta)), np.zeros(len(theta))
        for j in range(len(precomp['Tu'])):
            T = precomp['Tu'][j]['T']
            if j == 0:
                Kinv = linalg.inv(Kmax[:T, :T])
                logdet_K = fast_logdet(Kmax[:T, :T])
            else:
                # Here, we compute the inverse of K for the current
                # T from its known inverse for the previous T,
                # using block matrix inversion identities.
                Kinv, logdet_K = self._sym_block_inversion(
                    Kmax[:T, :T], Kinv, -logdet_K
                )
            Thalf = int(np.ceil(T / 2.0))
            mkr = int(np.ceil(0.5 * T ** 2))
            numTrials = precomp['Tu'][j]['numTrials']
            PautoSUM = precomp['Tu'][j]['PautoSUM'][i, :, :]

            for k, dKdtheta in enumerate(K_gradient.T):

                KinvM = Kinv[:Thalf, :].dot(dKdtheta[:T, :T])  # Thalf x T
                KinvMKinv = (KinvM.dot(Kinv)).T  # Thalf x T

                dg_KinvM = np.diag(KinvM)
                tr_KinvM = 2 * dg_KinvM.sum() - np.fmod(T, 2) * dg_KinvM[-1]

                pauto_kinv_dot = PautoSUM.ravel('F')[:mkr].dot(
                    KinvMKinv.ravel('F')[:mkr])
                pauto_kinv_dot_rest = PautoSUM.ravel('F')[-1:mkr - 1:- 1].dot(
                    KinvMKinv.ravel('F')[:(T ** 2 - mkr)])
                dEdtheta[k] = dEdtheta[k] - 0.5 * numTrials * tr_KinvM \
                    + 0.5 * pauto_kinv_dot \
                    + 0.5 * pauto_kinv_dot_rest

                f[i] = f[i] - 0.5 * numTrials * logdet_K \
                    - 0.5 * (PautoSUM * Kinv).sum()
        f_arr = -f
        df_arr = -dEdtheta

        return f_arr, df_arr

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
