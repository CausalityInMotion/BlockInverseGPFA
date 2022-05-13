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
from sklearn.decomposition import FactorAnalysis
from tqdm import trange
from . import gpfa_util


def fit(X, y_dim=3, bin_size=0.02, min_var_frac=0.01, em_tol=1.0E-8,
        em_max_iters=500, tau_init=0.1, eps_init=1.0E-3, freq_ll=5,
        verbose=False):
    """
    Fit the GPFA model with the given training data.

    Parameters
    ----------
    X   : numpy.ndarray
        training data structure containing np.ndarrays whose n-th
        element (corresponding to the n-th experimental trial) has
        shape of (#units, #bins)
    y_dim : int, optional
        state dimensionality
        Default: 3
    bin_size : float, optional
        spike bin width in sec
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
            gamma: np.ndarray of shape (1, #latent_vars)
                related to GP timescales by 'bin_size / sqrt(gamma)'
            eps: np.ndarray of shape (1, #latent_vars)
                GP noise variances
            d: np.ndarray of shape (#units, 1)
                observation mean
            C: np.ndarray of shape (#units, #latent_vars)
                mapping between the observed data space and the latent variable
                space
            R: np.ndarray of shape (#units, #latent_vars)
                observation noise covariance

    fit_info : dict
        Information of the fitting process and the parameters used there
        iteration_time : list
            containing the runtime for each iteration step in the EM algorithm.
    """
    # For compute efficiency, train on equal-length segments of trials
    x_train_cut = gpfa_util.cut_trials(X)
    if len(x_train_cut) == 0:
        warnings.warn('No segments extracted for training. Defaulting to '
                      'segLength=Inf.')
        x_train_cut = gpfa_util.cut_trials(X, seg_length=np.inf)

    # ==================================
    # Initialize state model parameters
    # ==================================
    params_init = {}
    params_init['covType'] = 'rbf'
    # GP timescale
    # Assume binWidth is the time step size.
    params_init['gamma'] = (bin_size / tau_init) ** 2 * np.ones(y_dim)
    # GP noise variance
    params_init['eps'] = eps_init * np.ones(y_dim)

    # ========================================
    # Initialize observation model parameters
    # ========================================
    print('Initializing parameters using factor analysis...')

    x_all = np.hstack(x_train_cut)
    fa = FactorAnalysis(n_components=y_dim, copy=True,
                        noise_variance_init=np.diag(np.cov(x_all, bias=True)))
    fa.fit(x_all.T)
    params_init['d'] = x_all.mean(axis=1)
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

    params_est, x_train_cut, ll_cut, iter_time = e_m(
        params_init, x_train_cut, min_var_frac=min_var_frac,
        max_iters=em_max_iters, tol=em_tol, freq_ll=freq_ll, verbose=verbose)

    fit_info = {'iteration_time': iter_time, 'log_likelihoods': ll_cut}

    return params_est, fit_info


def e_m(params_init, X, max_iters=500, tol=1.0E-8, min_var_frac=0.01,
        freq_ll=5, verbose=False):
    """
    Fits GPFA model parameters using expectation-maximization (EM) algorithm.

    Parameters
    ----------
    params_init : dict
        GPFA model parameters at which EM algorithm is initialized
        covType : {'rbf', 'tri', 'logexp'}
            type of GP covariance
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by
            'bin_size / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the observation data space and the
            latent variable space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance
    X : numpy.ndarray
        training data structure containing np.ndarrays whose n-th
        element (corresponding to the n-th experimental trial) has
        shape of (#units, #bins)
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
    seqs_latent : np.recarray
        a copy of the training data structure, augmented with the new
        fields:
        latent_variable : np.ndarray of shape (#latent_vars x #bins)
            posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
            posterior covariance between latent variables at each
            timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
            posterior covariance over time for each latent
            variable
    ll : list
        list of log likelihoods after each EM iteration
    iter_time : list
        lisf of computation times (in seconds) for each EM iteration
    """
    params = params_init
    t = np.array([t.shape[1] for t in X])
    _, y_dim = params['C'].shape
    lls = []
    ll_old = ll_base = ll = 0.0
    iter_time = []
    var_floor = min_var_frac * np.diag(np.cov(np.hstack(X)))
    seqs_latent = None

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
        seqs_latent, ll = exact_inference_with_ll(X, params,
                                                  get_ll=get_ll)
        lls.append(ll)

        # ==== M STEP ====
        sum_p_auto = np.zeros((y_dim, y_dim))
        for seq_latent in seqs_latent:
            sum_p_auto += seq_latent['Vsm'].sum(axis=2) \
                + seq_latent['latent_variable'].dot(
                seq_latent['latent_variable'].T)
        x = np.hstack(X)
        latent_variable = np.hstack(seqs_latent['latent_variable'])
        sum_xytrans = x.dot(latent_variable.T)
        sum_yall = latent_variable.sum(axis=1)[:, np.newaxis]
        sum_xall = x.sum(axis=1)[:, np.newaxis]

        # term is (xDim+1) x (xDim+1)
        term = np.vstack([np.hstack([sum_p_auto, sum_yall]),
                          np.hstack([sum_yall.T, t.sum().reshape((1, 1))])])
        # yDim x (xDim+1)
        cd = gpfa_util.rdiv(np.hstack([sum_xytrans, sum_xall]), term)

        params['C'] = cd[:, :y_dim]
        params['d'] = cd[:, -1]

        # yCent must be based on the new d
        # yCent = bsxfun(@minus, [seq.y], currentParams.d);
        # R = (yCent * yCent' - (yCent * [seq.latent_variable]') * \
        #     currentParams.C') / sum(T);
        c = params['C']
        d = params['d'][:, np.newaxis]
        if params['notes']['RforceDiagonal']:
            sum_xxtrans = (x * x).sum(axis=1)[:, np.newaxis]
            xd = sum_xall * d
            term = ((sum_xytrans - d.dot(sum_yall.T)) * c).sum(axis=1)
            term = term[:, np.newaxis]
            r = d ** 2 + (sum_xxtrans - 2 * xd - term) / t.sum()

            # Set minimum private variance
            r = np.maximum(var_floor, r)
            params['R'] = np.diag(r[:, 0])
        else:
            sum_xxtrans = x.dot(x.T)
            xd = sum_xall.dot(d.T)
            term = (sum_xytrans - d.dot(sum_yall.T)).dot(c.T)
            r = d.dot(d.T) + (sum_xxtrans - xd - xd.T - term) / t.sum()

            params['R'] = (r + r.T) / 2  # ensure symmetry

        if params['notes']['learnKernelParams']:
            res = learn_gp_params(seqs_latent, params, verbose=verbose)
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

    return params, seqs_latent, lls, iter_time


def exact_inference_with_ll(seqs_in, params, get_ll=True):
    """
    Extracts latent trajectories from observed data
    given GPFA model parameters.

    Parameters
    ----------
    seqs : np.recarray
        input data structure, whose n-th element (corresponding to the n-th
        experimental trial) of shape (#units, #bins)
    params : dict
        GPFA model parameters whe the following fields:
        C : np.ndarray
            FA factor loadings matrix
        d : np.ndarray
            FA mean vector
        R : np.ndarray
            FA noise covariance matrix
        gamma : np.ndarray
            GP timescale
        eps : np.ndarray
            GP noise variance
    get_ll : bool, optional
          specifies whether to compute data log likelihood (default: True)

    Returns
    -------
    seqs_latent : np.recarray
        seqs : np.recarray
            input data structure, whose n-th element (corresponding to the n-th
            experimental trial) has fields:
            x : np.ndarray of shape (#units, #bins)
        latent_variable :  (#latent_vars, #bins) np.ndarray
              posterior mean of latent variables at each time bin
        Vsm :  (#latent_vars, #latent_vars, #bins) np.ndarray
              posterior covariance between latent variables at each
              timepoint
        VsmGP :  (#bins, #bins, #latent_vars) np.ndarray
                posterior covariance over time for each latent
                variable
    ll : float
        data log likelihood, np.nan is returned when `get_ll` is set False
    """
    x_dim, y_dim = params['C'].shape

    # copy the contents of the input data structure to output structure
    seqs = np.empty(len(seqs_in), dtype=[('x', object)])
    for s, seq in enumerate(seqs):
        seq['x'] = seqs_in[s]

    dtype_out = [(x, seqs[x].dtype) for x in seqs.dtype.names]
    dtype_out.extend([('latent_variable', object), ('Vsm', object),
                      ('VsmGP', object)])
    seqs_latent = np.empty(len(seqs), dtype=dtype_out)
    for dtype_name in seqs.dtype.names:
        seqs_latent[dtype_name] = seqs[dtype_name]

    # Precomputations
    if params['notes']['RforceDiagonal']:
        rinv = np.diag(1.0 / np.diag(params['R']))
        logdet_r = (np.log(np.diag(params['R']))).sum()
    else:
        rinv = linalg.inv(params['R'])
        rinv = (rinv + rinv.T) / 2  # ensure symmetry
        logdet_r = gpfa_util.logdet(params['R'])

    c_rinv = params['C'].T.dot(rinv)
    c_rinv_c = c_rinv.dot(params['C'])

    t_all = [seqs[i][0].shape[1] for i in range(len(seqs))]
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
        c_rinv_c_big = linalg.block_diag(*blah)  # (xDim*T) x (xDim*T)
        minv, logdet_m = gpfa_util.inv_persymm(k_big_inv + c_rinv_c_big, y_dim)

        # Note that posterior covariance does not depend on observations,
        # so can compute once for all trials with same T.
        # xDim x xDim posterior covariance for each timepoint
        vsm = np.full((y_dim, y_dim, t), np.nan)
        idx = np.arange(0, y_dim * t + 1, y_dim)
        for i in range(t):
            vsm[:, :, i] = minv[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

        # T x T posterior covariance for each GP
        vsm_gp = np.full((t, t, y_dim), np.nan)
        for i in range(y_dim):
            vsm_gp[:, :, i] = minv[i::y_dim, i::y_dim]

        # Process all trials with length T
        n_list = np.where(t_all == t)[0]
        # dif is xDim x sum(T)
        dif = np.hstack(seqs_latent[n_list]['x']) - params['d'][:, np.newaxis]
        # term1Mat is (yDim*T) x length(nList)
        term1_mat = c_rinv.dot(dif).reshape((y_dim * t, -1), order='F')

        # Compute blkProd = CRinvC_big * invM efficiently
        # blkProd is block persymmetric, so just compute top half
        t_half = int(np.ceil(t / 2.0))
        blk_prod = np.zeros((y_dim * t_half, y_dim * t))
        idx = range(0, y_dim * t_half + 1, y_dim)
        for i in range(t_half):
            blk_prod[idx[i]:idx[i + 1], :] = c_rinv_c.dot(
                minv[idx[i]:idx[i + 1], :])
        blk_prod = k_big[:y_dim * t_half, :].dot(
            gpfa_util.fill_persymm(np.eye(y_dim * t_half, y_dim * t) -
                                   blk_prod, y_dim, t))
        # latent_variableMat is (yDim*T) x length(nList)
        latent_variable_mat = gpfa_util.fill_persymm(
            blk_prod, y_dim, t).dot(term1_mat)

        for i, n in enumerate(n_list):
            seqs_latent[n]['latent_variable'] = \
                latent_variable_mat[:, i].reshape((y_dim, t), order='F')
            seqs_latent[n]['Vsm'] = vsm
            seqs_latent[n]['VsmGP'] = vsm_gp

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

    return seqs_latent, ll


def learn_gp_params(seqs_latent, params, verbose=False):
    """Updates parameters of GP state model, given trajectories.

    Parameters
    ----------
    seqs_latent : np.recarray
        data structure containing trajectories;
    params : dict
        current GP state model parameters, which gives starting point
        for gradient optimization;
    verbose : bool, optional
        specifies whether to display status messages (default: False)

    Returns
    -------
    param_opt : np.ndarray
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

    y_dim = param_init.shape[-1]
    precomp = gpfa_util.make_precomp(seqs_latent, y_dim)

    # Loop once for each state dimension (each GP)
    for i in range(y_dim):
        const = {'eps': params['eps'][i]}
        initp = np.log(param_init[i])
        res_opt = optimize.minimize(gpfa_util.grad_betgam, initp,
                                    args=(precomp[i], const),
                                    method='L-BFGS-B', jac=True)
        param_opt['gamma'][i] = np.exp(res_opt.x)

        if verbose:
            print(f'\n Converged p; yDim:{i}, p:{res_opt.x}')

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
        gamma : np.ndarray of shape (1, #latent_vars)
            related to GP timescales by 'bin_size / sqrt(gamma)'
        eps : np.ndarray of shape (1, #latent_vars)
            GP noise variances
        d : np.ndarray of shape (#units, 1)
            observation mean
        C : np.ndarray of shape (#units, #latent_vars)
            mapping between the neuronal data space and the latent variable
            space
        R : np.ndarray of shape (#units, #latent_vars)
            observation noise covariance

    seqs : np.recarray
        Contains the embedding of the training data into the latent variable
        space.
        Data structure, whose n-th entry (corresponding to the n-th
        experimental trial) has field
        x : np.ndarray of shape (#units, #bins)
          observed data
        latent_variable : np.ndarray of shape (#latent_vars, #bins)
          posterior mean of latent variables at each time bin
        Vsm : np.ndarray of shape (#latent_vars, #latent_vars, #bins)
          posterior covariance between latent variables at each
          timepoint
        VsmGP : np.ndarray of shape (#bins, #bins, #latent_vars)
          posterior covariance over time for each latent variable

    Returns
    -------
    params_est : dict
        Estimated model parameters, including `Corth`, obtained by
        orthonormalizing the columns of C.
    seqs : np.recarray
        Training data structure that contains the new field
        `latent_variable_orth`, the orthonormalized neural trajectories.
    """
    C = params_est['C']
    Z = np.hstack(seqs['latent_variable'])
    latent_variable_orth, Corth, _ = gpfa_util.orthonormalize(Z, C)
    seqs = gpfa_util.segment_by_trial(
        seqs, latent_variable_orth, 'latent_variable_orth')

    params_est['Corth'] = Corth

    return Corth, seqs
