# ...
# Moving this here to prevent from showing up on sphinx pages
# Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
# Copyright 2014-2020 by the Elephant team.
# Modified BSD, see LICENSE.txt for details.
# ...

from __future__ import division, print_function, unicode_literals

from gpfa import GPFA
import numpy as np
from scipy import linalg
from scipy import sparse as sp

__all__ = [
    "GPFAInvPerSymm",
    "GPFANonInc"
]

class GPFAInvPerSymm(GPFA):
    """
    GPFA with Elephant's tridiagonal solver for block persymmetric kernel
    matrix inversion.
    """

    def _infer_latents(self, X, get_ll=True):
        """
        Inferrs latent trajectories from observed data given GPFA model
        parameters.
        Parameters
        ----------
        X   : an array-like, default=None
            An array-like sequence of time-series. The format is the same as
            for :meth:`fit``.
        get_ll : bool, optional, default=True
            specifies whether to compute data log likelihood
        Returns
        -------
        latent_seqs : numpy.recarray
            X_out : numpy.ndarray
                input data structure, whose n-th element (corresponding to the
                n-th experimental trial) has fields:
                X : numpy.ndarray of shape (x_dim, bins)
            Z_mu : (z_dim, bins) numpy.ndarray
                posterior mean of latent variables at each time bin
            Z_cov : (z_dim, z_dim, bins) numpy.ndarray
                posterior covariance between latent variables at each timepoint
            Z_covGP : (bins, bins, z_dim) numpy.ndarray
                    posterior covariance over time for each latent trajectory
        ll : float
            data log likelihood, returned when `get_ll` is set True
        """
        x_dim = self.C_.shape[0]

        # copy the contents of the input data structure to output structure
        X_out = np.empty(len(X), dtype=[('X', object)])
        for s, seq in enumerate(X_out):
            seq['X'] = X[s]

        dtype_out = [(i, X_out[i].dtype) for i in X_out.dtype.names]
        dtype_out.extend([('Z_mu', object), ('Z_cov', object),
                        ('Z_covGP', object)])
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
            logdet_r = self._logdet(self.R_)

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
            K_big_inv = linalg.inv(K_big[:t * self.z_dim, :t * self.z_dim])
            logdet_k_big = self._logdet(K_big[:t * self.z_dim, :t * self.z_dim])
            M = K_big_inv + C_rinv_c_big[:t * self.z_dim,:t * self.z_dim]
            M_inv, logdet_M = self._inv_persymm(M, self.z_dim)

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
            term1_mat = c_rinv.dot(dif).reshape(
                (self.z_dim * t, -1), order='F'
                )

            # latent_variable Matrix (Z_mat) is (z_dim*T) x length(nList)
            Z_mat = M_inv.dot(term1_mat)

            for i, n in enumerate(n_list):
                latent_seqs[n]['Z_mu'] = \
                    Z_mat[:, i].reshape((self.z_dim, t), order='F')
                latent_seqs[n]['Z_cov'] = vsm
                latent_seqs[n]['Z_covGP'] = vsm_gp

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

    def _grad_bet_theta(self, theta, gp_kernel_i, precomp, i):
        """
        Gradient computation for GP timescale optimization.
        This function is called by `_learn_gp_params()`
        Parameters
        ----------
        theta : numpy.array
            the flattened and log-transformed non-fixed hyperparams
            to which optimization is performed,
            where :math:`{\\theta} = log(\\text{kernel parameters})`
        gp_kernel_i : kernel instance
            the i-th GP kernel corresponding to the i-th latent trajectory
        precomp : numpy.recarray
            structure containing precomputations
        i : int
            The i-th index of the i-th latent trajectory
        Returns
        -------
        f : float
            values of objective function E[log P({x},{y})] at {\\theta}
        df_arr : numpy.array
            gradients at {\\theta}
        """
        gp_kernel_i.theta = theta
        Kmax, K_gradient = gp_kernel_i(
            precomp['Tsdt'], eval_gradient=True
            )
        dEdtheta = np.zeros(len(theta))
        f = 0.0
        for j in range(len(precomp['Tu'])):
            T = precomp['Tu'][j]['T']

            Kinv = linalg.inv(Kmax[:T, :T])
            logdet_K = self._logdet(Kmax[:T, :T])

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

            f = f - 0.5 * numTrials * logdet_K \
                - 0.5 * (PautoSUM * Kinv).sum()
        f = -f
        df_arr = -dEdtheta
        return f, df_arr
    
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

        res12 = self._rdiv(-term, F22)
        res11 = invA11 - res12.dot(term.T)
        res11 = (res11 + res11.T) / 2

        # Fill in bottom half of invM by picking elements from res11 and res12
        invM = self._fill_persymm(np.hstack([res11, res12]), blk_size, T)

        logdet_M = -self._logdet(invA11) + self._logdet(F22)

        return invM, logdet_M

    def _fill_persymm(self, p_in, blk_size, n_blocks, blk_size_vert=None):
        """
        Fills in the bottom half of a block persymmetric matrix, given the
        top half.
        Parameters
        ----------
        p_in :  (xDim*Thalf, xDim*T) np.ndarray
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
        Pout : (xDim*T, xDim*T) np.ndarray
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

    def _rdiv(self, a, b):
        """
        Returns the solution to x b = a. Equivalent to MATLAB right matrix
        division: a / b
        """
        return np.linalg.solve(b.T, a.T).T
    

class GPFANonInc(GPFA):
    """
    GPFA with non-incremental computation of the inverse of the
    kernel matrix using `numpy.linalg.inv`.
    """

    def _infer_latents(self, X, get_ll=True):
        """
        Inferrs latent trajectories from observed data given GPFA model
        parameters.
        Parameters
        ----------
        X   : an array-like, default=None
            An array-like sequence of time-series. The format is the same as
            for :meth:`fit``.
        get_ll : bool, optional, default=True
            specifies whether to compute data log likelihood
        Returns
        -------
        latent_seqs : numpy.recarray
            X_out : numpy.ndarray
                input data structure, whose n-th element (corresponding to the
                n-th experimental trial) has fields:
                X : numpy.ndarray of shape (x_dim, bins)
            Z_mu : (z_dim, bins) numpy.ndarray
                posterior mean of latent variables at each time bin
            Z_cov : (z_dim, z_dim, bins) numpy.ndarray
                posterior covariance between latent variables at each timepoint
            Z_covGP : (bins, bins, z_dim) numpy.ndarray
                    posterior covariance over time for each latent trajectory
        ll : float
            data log likelihood, returned when `get_ll` is set True
        """
        x_dim = self.C_.shape[0]

        # copy the contents of the input data structure to output structure
        X_out = np.empty(len(X), dtype=[('X', object)])
        for s, seq in enumerate(X_out):
            seq['X'] = X[s]

        dtype_out = [(i, X_out[i].dtype) for i in X_out.dtype.names]
        dtype_out.extend([('Z_mu', object), ('Z_cov', object),
                          ('Z_covGP', object)])
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
            logdet_r = self._logdet(self.R_)

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

            K_big_inv = linalg.inv(K_big[:t * self.z_dim, :t * self.z_dim])
            logdet_k_big = self._logdet(K_big[:t * self.z_dim, :t * self.z_dim])
            M = K_big_inv + C_rinv_c_big[:t * self.z_dim,:t * self.z_dim]
            M_inv = linalg.inv(M)
            logdet_M = self._logdet(M)

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
            term1_mat = c_rinv.dot(dif).reshape(
                (self.z_dim * t, -1), order='F'
                )

            # latent_variable Matrix (Z_mat) is (z_dim*T) x length(nList)
            Z_mat = M_inv.dot(term1_mat)

            for i, n in enumerate(n_list):
                latent_seqs[n]['Z_mu'] = \
                    Z_mat[:, i].reshape((self.z_dim, t), order='F')
                latent_seqs[n]['Z_cov'] = vsm
                latent_seqs[n]['Z_covGP'] = vsm_gp

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

    def _grad_bet_theta(self, theta, gp_kernel_i, precomp, i):
        """
        Gradient computation for GP timescale optimization.
        This function is called by `_learn_gp_params()`
        Parameters
        ----------
        theta : numpy.array
            the flattened and log-transformed non-fixed hyperparams
            to which optimization is performed,
            where :math:`{\\theta} = log(\\text{kernel parameters})`
        gp_kernel_i : kernel instance
            the i-th GP kernel corresponding to the i-th latent trajectory
        precomp : numpy.recarray
            structure containing precomputations
        i : int
            The i-th index of the i-th latent trajectory
        Returns
        -------
        f : float
            values of objective function E[log P({x},{y})] at {\\theta}
        df_arr : numpy.array
            gradients at {\\theta}
        """
        gp_kernel_i.theta = theta
        Kmax, K_gradient = gp_kernel_i(
            precomp['Tsdt'], eval_gradient=True
            )
        dEdtheta = np.zeros(len(theta))
        f = 0.0
        for j in range(len(precomp['Tu'])):
            T = precomp['Tu'][j]['T']

            Kinv = linalg.inv(Kmax[:T, :T])
            logdet_K = self._logdet(Kmax[:T, :T])

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

            f = f - 0.5 * numTrials * logdet_K \
                - 0.5 * (PautoSUM * Kinv).sum()
        f = -f
        df_arr = -dEdtheta
        return f, df_arr
