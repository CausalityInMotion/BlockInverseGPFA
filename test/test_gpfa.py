"""
GPFA Unittests.
:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
from scipy import linalg
from sklearn.decomposition import FactorAnalysis
from gpfa import GPFA


class TestGPFA(unittest.TestCase):
    """
    Unit tests for the GPFA analysis.
    """
    def setUp(self):
        """
        Set up synthetic data, initial parameters to help with the
        functions to be tested
        """
        np.random.seed(0)
        self.bin_size = 0.02  # [s]
        self.tau_init = 0.1  # [s]
        self.eps_init = 1.0E-3
        self.n_iters = 10
        self.z_dim = 2
        self.n_neurons = 2  # per rate therefore, there are 4 neurons

        def gen_test_data(trial_lens, rates_a, rates_b, use_sqrt=True):
            """
            Generate test data
            There are 2 x number of neurons for each group -- the first 2
            neurons use rates from set a, and the second 2 neurons use rates
            from set b.
            Args:
                trial_lens  : list of durations of each trial in [s]
                            len(trial_lens) corresponds with num of trials
                rates_a     : list of rates, one for each different time epoch.
                            Each each is a quarter of the total length.
                rates_b     : list of rates, one for each different time epoch
                            shuffled differently from rates_a
                n_neurons   : number of neurons
                bin_size    : bin size in [s] for analysis purpose
                use_sqrt    : boolean
                            if true, take square root of binned spike trains
            Returns:
                seqs        : an array-like of binned spiketrains arrays per
                            trial
            """
            # check the length of rates_a and rates_b must both be equal to 4
            if len(rates_a) != 4:
                raise ValueError("'rates_a' must have 4 elements in it")
            if len(rates_b) != 4:
                raise ValueError("'rates_b' must have 4 elements in it")

            seqs = np.empty(len(trial_lens), object)

            # generate data where num trials is len(trial_lens)
            for n, t_l in enumerate(trial_lens):

                # get number of bins for the each epoch
                # each each is a quarter of the total length.
                epoch_len = int(t_l / len(rates_a))
                nbins_per_epoch = int(epoch_len / self.bin_size)

                # generate two spike trains each with two neurons
                # neurons one and two use rates_a
                # neuros three and four use rates_b
                # concatenate them into one spiketrain
                spk_rates_a = np.random.poisson(
                            rates_a[0], (self.n_neurons, nbins_per_epoch))
                spk_rates_b = np.random.poisson(
                            rates_b[0], (self.n_neurons, nbins_per_epoch))
                binned_spikecount = np.concatenate([spk_rates_a, spk_rates_b])

                l_rates_a = len(rates_a)

                # loop over the remaining rates
                for i in range(1, l_rates_a):
                    # get number of bins for the remaining epochs
                    # n_bins_per_dur = int(durs[i] / bin_size)
                    spk_rates_a = np.random.poisson(
                            rates_a[i], (self.n_neurons, nbins_per_epoch))
                    spk_rates_b = np.random.poisson(
                            rates_b[i], (self.n_neurons, nbins_per_epoch))
                    spk_i = np.concatenate([spk_rates_a, spk_rates_b])
                    # concatenate previous spiketrains with new spiketrains
                    # from current duration
                    binned_spikecount = np.concatenate(
                                        [binned_spikecount, spk_i], axis=1)
                    # take square root of the binned_spikeCount
                    # if `use_sqrt` is True (see paper for motivation)
                    if use_sqrt:
                        binned_sqrt_spkcount = np.sqrt(binned_spikecount)

                    seqs[n] = binned_sqrt_spkcount

            return seqs

        rates_a = (2, 10, 2, 2)
        rates_b = (2, 2, 10, 2)
        trial_lens = [8, 10]

        # generate data
        self.X = gen_test_data(trial_lens, rates_a, rates_b)

        # get the number of time steps for each trial
        self.T = np.array([X_n.shape[1] for X_n in self.X])
        self.t_half = int(np.ceil(self.T[0] / 2.0))

        # Initialize state model parameters
        self.params_init = {}
        self.params_init['covType'] = 'rbf'
        # GP timescale
        # Assume binWidth is the time step size.
        time_step = (self.bin_size / self.tau_init) ** 2
        self.params_init['gamma'] = time_step * np.ones(self.z_dim)
        # GP noise variance
        self.params_init['eps'] = self.eps_init * np.ones(self.z_dim)

        # Initialize observation model parameters using factor analysis
        X_all = np.hstack([self.X[0]])
        f_a = FactorAnalysis(
            n_components=self.z_dim, copy=True,
            noise_variance_init=np.diag(np.cov(X_all, bias=True))
                            )
        # fit factor analysis
        f_a.fit(X_all.T)
        self.params_init['d'] = X_all.mean(axis=1)
        self.params_init['C'] = f_a.components_.T
        self.params_init['R'] = np.diag(f_a.noise_variance_)

        # Define parameter constraints
        self.params_init['notes'] = {
            'learnKernelParams': True,
            'learnGPNoise': False,
            'RforceDiagonal': True,
        }

        self.gpfa = GPFA(
            bin_size=self.bin_size, z_dim=self.z_dim,
            em_max_iters=self.n_iters
            )
        self.gpfa.fit(self.X)
        self.results, _ = self.gpfa.predict(
                                returned_data=['pZ_mu', 'pZ_mu_orth'])

    def test_infer_latents(self):
        """
        Test the GPFA mean and covariance using the equation
        A5 from the Byron et a,. (2009) paper since the
        implementation is different from equation A5.
        Here mean is defined by (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
        and covaraince is (K_inv + C'R_invC)
        """
        test_latent_seqs = np.empty(
            len(self.X), dtype=[('pZ_mu', object), ('pZ_cov', object)])

        for n, t in enumerate(self.T):
            # get the kernal as defined in GPFA
            _, k_big_inv, _ = self.gpfa._make_k_big(
                                                    params=self.params_init,
                                                    n_timesteps=t)
            rinv = np.diag(1.0 / np.diag(self.params_init['R']))
            c_rinv = self.params_init['C'].T.dot(rinv)

            # C'R_invC
            c_rinv_c = c_rinv.dot(self.params_init['C'])

            # subtract mean from activities (y - d)
            dif = np.hstack([self.X[n]]) - \
                self.params_init['d'][:, np.newaxis]
            # C'R_inv * (y - d)
            term1_mat = c_rinv.dot(dif).reshape(
                                        (self.z_dim * t, -1), order='F')
            # make a c_rinv_c big and block diagonal
            blah = [c_rinv_c for _ in range(t)]
            c_rinv_c_big = linalg.block_diag(*blah)  # (x_dim*T) x (x_dim*T)

            # (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
            test_latent_seqs[n]['pZ_mu'] = linalg.inv(
                k_big_inv + c_rinv_c_big).dot(term1_mat).reshape(
                    (self.z_dim, t), order='F')

            # compute covariance
            cov = np.full((self.z_dim, self.z_dim, t), np.nan)
            idx = np.arange(0, self.z_dim * t + 1, self.z_dim)
            for i in range(t):
                cov[:, :, i] = linalg.inv(
                    k_big_inv + c_rinv_c_big)[
                        idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

            test_latent_seqs[n]['pZ_cov'] = cov
        # get mean and covariance as implemented by GPFA
        latent_seqs, _ = self.gpfa._infer_latents(
                                        self.X, self.params_init
            )
        # Assert
        self.assertTrue(np.allclose(
                latent_seqs['pZ_mu'][0],
                test_latent_seqs['pZ_mu'][0]))
        self.assertTrue(np.allclose(
                latent_seqs['pZ_cov'][0],
                test_latent_seqs['pZ_cov'][0]))

    def test_fill_persymm(self):
        """
        GPFA takes advantage of the persymmetric structure of k_big_inv
        by only computing the top half of the metric and filling the bottom
        half with results from the top half.
        Test if fill_persymm returns an expected filled matrix
        """
        _, k_big_inv, _ = self.gpfa._make_k_big(
                                                self.params_init,
                                                self.T[0]
                                                )
        full_k_big_inv = self.gpfa._fill_persymm(
                                k_big_inv[:(self.z_dim*self.t_half), :],
                                self.z_dim, self.T[0])
        # Assert
        self.assertTrue(np.allclose(k_big_inv, full_k_big_inv))

    def test_orthonormalized_transform(self):
        """
        Test GPFA orthonormalization transform of the parameter `C`.
        """
        corth = self.gpfa.params_estimated['Corth']
        c_orth = linalg.orth(self.gpfa.params_estimated['C'])
        # Assert
        self.assertTrue(np.allclose(c_orth, corth))

    def test_orthonormalized_latents(self):
        """
        Test GPFA orthonormalization functions applied in `gpfa.predict`.
        """
        pZ_mu = self.results['pZ_mu']
        pZ_mu_orth = self.results['pZ_mu_orth']
        Z_all = np.hstack(pZ_mu)
        test_pZ_mu_orth = np.dot(self.gpfa.OrthTrans, Z_all)
        # get the right format of test_pZ_mu_orth
        test_seqs = self.gpfa._segment_by_trial(
            self.gpfa.train_latent_seqs,
            test_pZ_mu_orth,
            'test_pZ_mu_orth'
        )
        # Assert
        self.assertTrue(np.allclose(pZ_mu_orth[0],
                                    test_seqs['test_pZ_mu_orth'][0]))
