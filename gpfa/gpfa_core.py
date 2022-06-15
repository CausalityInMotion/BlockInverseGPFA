"""
GPFA core functionality.

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
from sklearn.utils.extmath import fast_logdet
from sklearn.decomposition import FactorAnalysis
from tqdm import trange
from . import gpfa_util


def fit(X, z_dim=3, bin_size=0.02, min_var_frac=0.01, em_tol=1.0E-8,
        em_max_iters=500, tau_init=0.1, eps_init=1.0E-3, freq_ll=5,
        verbose=False):
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
        Default: 0.1 [s]
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
                mapping between the observed data space and the latent variable
                space
            R: numpy.ndarray of shape (#x_dim, #z_dim)
                observation noise covariance

    fit_info : dict
        Information of the fitting process and the parameters used there
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
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
    fa = FactorAnalysis(n_components=z_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(X_all, bias=True)))
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

    params_est, X, ll_cut, iter_time = em(
        params_init, X, min_var_frac=min_var_frac,
        max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll, verbose=verbose)

    fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

    return params_est, fit_info


def em(params_init, X, max_iters=500, tol=1.0E-8, min_var_frac=0.01,
       freq_ll=5, verbose=False):
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

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
        fraction of overall data variance for each observed dimension to set as
        the private variance floor.  This is used to combat Heywood cases,
        where ML parameter learning returns one or more zero private variances.
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
        latent_seqs, ll = infer_latents(X, params, get_ll=get_ll)
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
            res = learn_gp_params(latent_seqs, params, verbose=verbose)
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


def infer_latents(X, params, get_ll=True):
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
        k_big, k_big_inv, logdet_k_big = gpfa_util.make_k_big(params, t)
        k_big = sparse.csr_matrix(k_big)

        blah = [c_rinv_c for _ in range(t)]
        c_rinv_c_big = linalg.block_diag(*blah)  # (x_dim*T) x (x_dim*T)
        minv, logdet_m = gpfa_util.inv_persymm(k_big_inv + c_rinv_c_big, z_dim)

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
            gpfa_util.fill_persymm(np.eye(z_dim * t_half, z_dim * t) -
                                   blk_prod, z_dim, t))
        # latent_variable Matrix (Z_mat) is (z_dim*T) x length(nList)
        Z_mat = gpfa_util.fill_persymm(
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


def learn_gp_params(latent_seqs, params, verbose=False):
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
    precomp = gpfa_util.make_precomp(latent_seqs, z_dim)

    # Loop once for each state dimension (each GP)
    for i in range(z_dim):
        const = {'eps': params['eps'][i]}
        initp = np.log(param_init[i])
        res_opt = optimize.minimize(gpfa_util.grad_betgam, initp,
                                    args=(precomp[i], const),
                                    method='L-BFGS-B', jac=True)
        param_opt['gamma'][i] = np.exp(res_opt.x)

        if verbose:
            print(f'\n Converged p; z_dim:{i}, p:{res_opt.x}')

    return param_opt


def orthonormalize(params_est, seqs):
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
    pZ_mu_orth, Corth, _ = gpfa_util.orthonormalize(Z_all, C)
    seqs = gpfa_util.segment_by_trial(seqs, pZ_mu_orth, 'pZ_mu_orth')

    params_est['Corth'] = Corth

    return Corth, seqs
