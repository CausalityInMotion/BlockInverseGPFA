"""
GPFA Unittests.
:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:copyright: Copyright 2014-2020 by the Elephant team.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
from scipy import linalg
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
        np.random.seed(10)
        self.bin_size = 0.02  # [s]
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
                # each epoch is a quarter of the total length.
                epoch_len = int(np.ceil(t_l / len(rates_a)))
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

        # initialize GPFA
        self.gpfa = GPFA(
            bin_size=self.bin_size, z_dim=self.z_dim,
            em_max_iters=self.n_iters
            )
        # fit the model
        self.gpfa.fit(self.X)
        self.results, _ = self.gpfa.predict(
                                returned_data=['pZ_mu', 'pZ_mu_orth'])

        # get latents sequence and data log_likelihood
        self.latent_seqs, self.ll = self.gpfa._infer_latents(self.X)

    # @unittest.skip("skipping")
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
            k_big = self.gpfa._make_k_big(n_timesteps=t)
            k_big_inv = linalg.inv(k_big)
            rinv = np.diag(1.0 / np.diag(self.gpfa.R_))
            c_rinv = self.gpfa.C_.T.dot(rinv)

            # C'R_invC
            c_rinv_c = c_rinv.dot(self.gpfa.C_)

            # subtract mean from activities (y - d)
            dif = np.hstack([self.X[n]]) - \
                self.gpfa.d_[:, np.newaxis]
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
        latent_seqs = self.latent_seqs
        # Assert
        self.assertTrue(np.allclose(
                latent_seqs['pZ_mu'][0],
                test_latent_seqs['pZ_mu'][0]))
        self.assertTrue(np.allclose(
                latent_seqs['pZ_cov'][0],
                test_latent_seqs['pZ_cov'][0]))

    def test_data_loglikelihood(self):
        """
        Test the data log_likelihood
        """
        test_ll = -4092.076151443287
        ll = self.ll
        # Assert
        self.assertEqual(test_ll, ll)

    # @unittest.skip("skipping")
    def test_orthonormalized_transform(self):
        """
        Test GPFA orthonormalization transform of the parameter `C`.
        """
        corth = self.gpfa.Corth_
        c_orth = linalg.orth(self.gpfa.C_)
        # Assert
        self.assertTrue(np.allclose(c_orth, corth))

    # @unittest.skip("skipping")
    def test_orthonormalized_latents(self):
        """
        Test GPFA orthonormalization functions applied in `gpfa.predict`.
        """
        pZ_mu = self.results['pZ_mu'][0]
        pZ_mu_orth = self.results['pZ_mu_orth'][0]
        test_pZ_mu_orth = np.dot(self.gpfa.OrthTrans_, pZ_mu)
        # Assert
        self.assertTrue(np.allclose(pZ_mu_orth, test_pZ_mu_orth))
