# ...
# Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
# license Modified BSD, see LICENSE.txt for details.
# ...
"""
BlockInvGPFA Unittests
"""

import unittest
import numpy as np
from scipy import linalg
from blockinvgpfa import BlockInvGPFA
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


class TestGPFA(unittest.TestCase):
    """
    Unit tests for the BlockInvGPFA analysis.
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

        # ==================================================
        # generate data
        # ==================================================
        rates_a = (2, 10, 2, 2)
        rates_b = (2, 2, 10, 2)
        trial_lens = [8, 10]

        self.X = gen_test_data(trial_lens, rates_a, rates_b)

        # get the number of time steps for each trial
        self.T = np.array([X_n.shape[1] for X_n in self.X])
        self.t_half = int(np.ceil(self.T[0] / 2.0))

        # ==================================================
        # initialize BlockInvGPFA
        # ==================================================
        multi_params_kernel = ConstantKernel(
                        1-0.001, constant_value_bounds='fixed'
                        ) * RBF(length_scale=0.1) + ConstantKernel(
                            0.001, constant_value_bounds='fixed'
                            ) * WhiteKernel(
                                noise_level=1
                                )
        seq_kernel = [ConstantKernel(
                        1-0.001, constant_value_bounds='fixed'
                        ) * RBF(length_scale=0.1) + ConstantKernel(
                            0.001, constant_value_bounds='fixed'
                            ) * WhiteKernel(
                                noise_level=1, noise_level_bounds='fixed'
                                ),
                      ConstantKernel(
                        1-0.001, constant_value_bounds='fixed'
                        ) * RBF(length_scale=0.1) + ConstantKernel(
                            0.001, constant_value_bounds='fixed'
                            ) * WhiteKernel(
                                noise_level=1, noise_level_bounds='fixed'
                                    )]
        self.blockinv_gpfa = BlockInvGPFA(
            bin_size=self.bin_size, z_dim=self.z_dim,
            em_max_iters=self.n_iters
            )
        self.gpfa_with_seq_kernel = BlockInvGPFA(
            bin_size=self.bin_size, z_dim=self.z_dim,
            gp_kernel=seq_kernel,
            em_max_iters=self.n_iters
            )
        self.gpfa_with_multi_params_kernel = BlockInvGPFA(
            bin_size=self.bin_size, z_dim=self.z_dim,
            gp_kernel=multi_params_kernel,
            em_max_iters=self.n_iters
            )
        # fit the model
        self.blockinv_gpfa.fit(self.X)
        self.gpfa_with_multi_params_kernel.fit(self.X)
        self.gpfa_with_seq_kernel.fit(self.X)
        self.results, _ = self.blockinv_gpfa.predict(
                                returned_data=['Z_mu', 'Z_mu_orth'])

        # get latents sequence and data log_likelihood
        self.latent_seqs, self.ll = self.blockinv_gpfa._infer_latents(self.X)
        self.latent_seqs_multiparamskern, self.ll_multiparams_kernel = \
            self.gpfa_with_multi_params_kernel._infer_latents(self.X)
        self.latent_seqs_seqkernel, self.ll_seq_kernel = \
            self.gpfa_with_seq_kernel._infer_latents(self.X)

    def create_mu_and_cov(self, gpfa_inst):
        """
        Create the GPFA mean and covariance using the equation
        A5 from the Byron et a,. (2009) paper since the
        implementation is different from equation A5. Here mean is
        defined by (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
        and covaraince is (K_inv + C'R_invC)

        Paramters:
        gpfa_inst : BlockInvGPFA instance
            Each istance is different based on the input params
        Returns:
        test_latent_seqs: numpy.ndarray
            GPFA mean and cov
        """
        test_latent_seqs = np.empty(
            len(self.X), dtype=[('Z_mu', object), ('Z_cov', object)])

        for n, t in enumerate(self.T):
            # get the kernal as defined in GPFA
            k_big = gpfa_inst._make_k_big(n_timesteps=t)
            k_big_inv = linalg.inv(k_big)
            rinv = np.diag(1.0 / np.diag(gpfa_inst.R_))
            c_rinv = gpfa_inst.C_.T.dot(rinv)

            # C'R_invC
            c_rinv_c = c_rinv.dot(gpfa_inst.C_)

            # subtract mean from activities (y - d)
            dif = np.hstack([self.X[n]]) - \
                gpfa_inst.d_[:, np.newaxis]
            # C'R_inv * (y - d)
            term1_mat = c_rinv.dot(dif).reshape(
                                        (self.z_dim * t, -1), order='F')
            # make a c_rinv_c big and block diagonal
            blah = [c_rinv_c for _ in range(t)]
            c_rinv_c_big = linalg.block_diag(*blah)  # (x_dim*T) x (x_dim*T)

            # (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
            test_latent_seqs[n]['Z_mu'] = linalg.inv(
                k_big_inv + c_rinv_c_big).dot(term1_mat).reshape(
                    (self.z_dim, t), order='F')

            # compute covariance
            cov = np.full((self.z_dim, self.z_dim, t), np.nan)
            idx = np.arange(0, self.z_dim * t + 1, self.z_dim)
            for i in range(t):
                cov[:, :, i] = linalg.inv(
                    k_big_inv + c_rinv_c_big)[
                        idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

            test_latent_seqs[n]['Z_cov'] = cov
            return test_latent_seqs

    def test_infer_latents(self):
        """
        Test the mean and cov for different BlockInvGPFA instances
        """
        # get test mean and cov for different BlockInvGPFA instances
        test_latent_seqs_gpfa = self.create_mu_and_cov(self.blockinv_gpfa)
        test_latent_seqs_seq_kern = self.create_mu_and_cov(
            self.gpfa_with_seq_kernel
        )
        test_latent_seqs_multiparams = self.create_mu_and_cov(
            self.gpfa_with_multi_params_kernel
        )
        # Assert
        self.assertTrue(np.allclose(
                self.latent_seqs['Z_mu'][0],
                test_latent_seqs_gpfa['Z_mu'][0]))
        self.assertTrue(np.allclose(
                self.latent_seqs['Z_cov'][0],
                test_latent_seqs_gpfa['Z_cov'][0]))
        self.assertTrue(np.allclose(
                self.latent_seqs_seqkernel['Z_mu'][0],
                test_latent_seqs_seq_kern['Z_mu'][0]))
        self.assertTrue(np.allclose(
                self.latent_seqs_seqkernel['Z_cov'][0],
                test_latent_seqs_seq_kern['Z_cov'][0]))
        self.assertTrue(np.allclose(
                self.latent_seqs_multiparamskern['Z_mu'][0],
                test_latent_seqs_multiparams['Z_mu'][0]))
        self.assertTrue(np.allclose(
                self.latent_seqs_multiparamskern['Z_cov'][0],
                test_latent_seqs_multiparams['Z_cov'][0]))

    def test_data_loglikelihood(self):
        """
        Test the data log_likelihood
        """
        test_ll = -4092.076117337763
        # Assert
        self.assertAlmostEqual(test_ll, self.ll)
        self.assertAlmostEqual(test_ll, self.ll_seq_kernel)
        self.assertGreater(self.ll_multiparams_kernel, test_ll)

    def test_orthonormalized_transform(self):
        """
        Test BlockInvGPFA orthonormalization transform of the parameter `C`.
        """
        corth = self.blockinv_gpfa.Corth_
        c_orth = linalg.orth(self.blockinv_gpfa.C_)
        # Assert
        self.assertTrue(np.allclose(c_orth, corth))

    def test_orthonormalized_latents(self):
        """
        Test BlockInvGPFA orthonormalization functions applied in
        `blockinv_gpfa.predict`.
        """
        Z_mu = self.results['Z_mu'][0]
        Z_mu_orth = self.results['Z_mu_orth'][0]
        test_Z_mu_orth = np.dot(self.blockinv_gpfa.OrthTrans_, Z_mu)
        # Assert
        self.assertTrue(np.allclose(Z_mu_orth, test_Z_mu_orth))

    def test_variance_explained(self):
        """
        Test BlockInvGPFA explained_variance
        """
        test_r2_score = 0.6648115733320232
        r2_t1 = self.blockinv_gpfa.variance_explained()[0]
        # Assert
        self.assertAlmostEqual(test_r2_score, r2_t1)
