"""
GPFA util functions.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function, unicode_literals

import warnings
import numpy as np
import scipy as sp


def cut_trials(X_in, seg_length=20):
    """
    Extracts trial segments that are all of the same length.  Uses
    overlapping segments if trial length is not integer multiple
    of segment length.  Ignores trials with length shorter than
    one segment length.

    Parameters
    ----------
    X_in : a list of observation sequences, one per trial.
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


def logdet(A):
    """
    log(det(A)) where A is positive-definite.
    This is faster and more stable than using log(det(A)).

    Written by Tom Minka
    (c) Microsoft Corporation. All rights reserved.
    """
    U = np.linalg.cholesky(A)
    return 2 * (np.log(np.diag(U))).sum()


def make_k_big(params, n_timesteps):
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
        represents the covariance between the state vectors at timesteps t1 and
        t2. K_big is sparse and striped.
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
        # the original MATLAB program uses here a special algorithm, provided
        # in C and MEX, for inversion of Toeplitz matrix:
        # [K_big_inv(idx+i, idx+i), logdet_K] = invToeplitz(K);
        # TO-DO: use an inversion method optimized for Toeplitz matrix
        # Below is an attempt to use such a method, not leading to a speed-up.
        # # K_big_inv[i::x_dim, i::x_dim] = sp.linalg.solve_toeplitz((K[:, 0],
        # K[0, :]), np.eye(T))
        K_big_inv[i::z_dim, i::z_dim] = np.linalg.inv(K)
        logdet_K = logdet(K)

        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big


def inv_persymm(M, blk_size):
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
    invM = fill_persymm(np.hstack([res11, res12]), blk_size, T)

    logdet_M = -logdet(invA11) + logdet(F22)

    return invM, logdet_M


def fill_persymm(p_in, blk_size, n_blocks, blk_size_vert=None):
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


def make_precomp(Seqs, z_dim):
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


def grad_betgam(p, pre_comp, const):
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
        logdet_K = logdet(Kmax[:T, :T])

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


def orthonormalize(Z, l_mat):
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


def segment_by_trial(seqs, Z, fn):
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
