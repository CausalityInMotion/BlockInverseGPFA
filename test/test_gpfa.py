"""
GPFA Unittests.

:copyright: Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
:license: Modified BSD, see LICENSE.txt for details.
"""

import unittest
import numpy as np
from scipy import linalg
from sklearn.decomposition import FactorAnalysis
from gpfa import GPFA, gpfa_core, gpfa_util


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
        self.n_trials = 1
        self.bin_size = 0.02  # [s]
        self.tau_init = 0.1  # [s]
        self.eps_init = 1.0E-3
        self.n_iters = 10
        self.x_dim = 2
        n_neurons = 2  # per rate therefore, there are 4 neurons

        def gen_test_data(rates_a, rates_b, durs, n_neurons,
                          bin_size, use_sqrt=True):
            """
            Generate test data
            There are 2 x n_neuron neurons -- the first n_neuron
            neurons use rates from set a, and the second n_neuron
            neurons use rates from set b.
            Args:
                rates_a     : list of rates, one for each different time epoch
                rates_b     : list of rates, one for each different time epoch
                            shuffled differently from rates_a
                durs        : list of durations of each time epoch in [s]
                n_neurons   : number of neurons
                bin_size    : bin size in [s] for analysis purpose
                use_sqrt    : boolean
                            if true, take square root of binned spike trains

            Returns:
                seqs        : a list of binned spiketrains arrays per trial

            """
            # get number of bins for the first epoch
            # for both rates_a and rates_b
            n_bins_per_dur = int(durs[0] / bin_size)

            # generate two spike trains each with two neurons
            # neurons one and two use rates_a
            # neuros three and four use rates_b
            # concatenate them into one spiketrain
            spk_rates_a = np.random.poisson(rates_a[0],
                                            (n_neurons, n_bins_per_dur))
            spk_rates_b = np.random.poisson(rates_b[0],
                                            (n_neurons, n_bins_per_dur))
            binned_spikecount = np.concatenate([spk_rates_a, spk_rates_b])

            l_rates_a = len(rates_a)

            # loop over the remaining rates
            for i in range(1, l_rates_a):
                # get number of bins for the remaining epochs
                n_bins_per_dur = int(durs[i] / bin_size)
                spk_rates_a = np.random.poisson(rates_a[i],
                                                (n_neurons, n_bins_per_dur))
                spk_rates_b = np.random.poisson(rates_b[i],
                                                (n_neurons, n_bins_per_dur))
                spk_i = np.concatenate([spk_rates_a, spk_rates_b])
                # concatenate previous spiketrains with new spiketrains
                # from current duration
                binned_spikecount = np.concatenate(
                                                [binned_spikecount,
                                                 spk_i], axis=1
                                                )

            # get number of bins
            n_bins = binned_spikecount.shape[1]

            # take square root of the binned_spikeCount
            # if `use_sqrt` is True (see paper for motivation)
            if use_sqrt:
                binned_sqrt_spkcount = np.sqrt(binned_spikecount)

            seqs = [(n_bins, binned_sqrt_spkcount)]

            return seqs

        rates_a = (2, 10, 2, 2)
        rates_b = (2, 2, 10, 2)
        durs = (2.5, 2.5, 2.5, 2.5)

        # covert generated data to sequence spiketrains
        seqs = gen_test_data(rates_a, rates_b, durs,
                             n_neurons, bin_size=self.bin_size)

        # add fields to the np.array to make it a np.recarray
        self.data = np.array(seqs, dtype=[('T', int), ('y', 'O')])

        # get the number of time steps in the trails
        self.t_all = self.data['T'][0]
        self.t_half = int(np.ceil(self.t_all / 2.0))

        # Initialize state model parameters
        self.params_init = {}
        self.params_init['covType'] = 'rbf'
        # GP timescale
        # Assume binWidth is the time step size.
        time_step = (self.bin_size / self.tau_init) ** 2
        self.params_init['gamma'] = time_step * np.ones(self.x_dim)
        # GP noise variance
        self.params_init['eps'] = self.eps_init * np.ones(self.x_dim)

        # Initialize observation model parameters using factor analysis
        y_all = np.hstack(self.data['y'])
        f_a = FactorAnalysis(
            n_components=self.x_dim, copy=True,
            noise_variance_init=np.diag(np.cov(y_all, bias=True))
                            )
        # fit factor analysis
        f_a.fit(y_all.T)
        self.params_init['d'] = y_all.mean(axis=1)
        self.params_init['C'] = f_a.components_.T
        self.params_init['R'] = np.diag(f_a.noise_variance_)

        # Define parameter constraints
        self.params_init['notes'] = {
            'learnKernelParams': True,
            'learnGPNoise': False,
            'RforceDiagonal': True,
        }

    def test_fit_transform(self):
        """
        Test the fit and tranform methods
        against the tif_transform
        """
        gpfa1 = GPFA(
            bin_size=self.bin_size, x_dim=self.x_dim,
            em_max_iters=self.n_iters
            )
        gpfa1.fit(self.data)
        latent_variable_orth1 = gpfa1.transform(self.data)
        latent_variable_orth2 = GPFA(
            bin_size=self.bin_size, x_dim=self.x_dim,
            em_max_iters=self.n_iters).fit_transform(self.data)
        for i in range(len(self.data)):
            for j in range(self.x_dim):
                self.assertTrue(
                    np.allclose(
                        latent_variable_orth1[i][j],
                        latent_variable_orth2[i][j]
                        )
                    )

    def test_exact_inference_with_ll(self):
        """
        Test the GPFA mean and covariance using the equation
        A5 from the Byron et a,. (2009) paper since the
        implementation is different from equation A5.

        Equation A5 can only be implemented for 1 trial hence, n_trial == 1

        Here mean is defined by (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
        and covaraince is (K_inv + C'R_invC)
        """
        # get the kernal as defined in GPFA
        _, k_big_inv, _ = gpfa_util.make_k_big(
                                                self.params_init,
                                                self.t_all
                                            )
        rinv = np.diag(1.0 / np.diag(self.params_init['R']))
        c_rinv = self.params_init['C'].T.dot(rinv)

        # C'R_invC
        c_rinv_c = c_rinv.dot(self.params_init['C'])

        # subtract mean from activities (y - d)
        dif = np.hstack(self.data['y']) - \
            self.params_init['d'][:, np.newaxis]
        # C'R_inv * (y - d)
        term1_mat = c_rinv.dot(dif).reshape(
                                    (self.x_dim * self.t_all, -1),
                                    order='F'
            )
        # make a c_rinv_c big and block diagonal
        blah = [c_rinv_c for _ in range(self.t_all)]
        c_rinv_c_big = linalg.block_diag(*blah)

        # (K_inv + C'R_invC)^-1 * C'R_inv * (y - d)
        latent_var = linalg.inv(
            k_big_inv + c_rinv_c_big).dot(term1_mat).reshape(
                (self.x_dim, self.t_all),
                order='F'
                )

        # compute covariance
        cov = np.full((self.x_dim, self.x_dim, self.t_all), np.nan)
        idx = np.arange(0, self.x_dim * self.t_all + 1, self.x_dim)
        for i in range(self.t_all):
            cov[:, :, i] = linalg.inv(
                k_big_inv + c_rinv_c_big)[idx[i]:idx[i + 1], idx[i]:idx[i + 1]]

        # get mean and covariance as implemented by GPFA
        seqs_latent, _ = gpfa_core.exact_inference_with_ll(
            self.data, self.params_init
            )
        # Assert
        self.assertTrue(np.allclose(
            seqs_latent['latent_variable'][0][0],
            latent_var[0]))
        self.assertTrue(np.allclose(seqs_latent['Vsm'][0][0][0], cov[0][0]))

    def test_fill_persymm(self):
        """
        GPFA takes advantage of the persymmetric structure of k_big_inv
        by only computing the top half of the metric and filling the bottom
        half with results from the top half.

        Test if fill_persymm returns an expected filled matrix
        """
        _, k_big_inv, _ = gpfa_util.make_k_big(
                                                self.params_init,
                                                self.t_all
                                                )
        full_k_big_inv = gpfa_util.fill_persymm(
                                k_big_inv[:(self.x_dim*self.t_half), :],
                                self.x_dim, self.t_all)
        # Assert
        self.assertTrue(np.allclose(k_big_inv, full_k_big_inv))

    def test_orthonormalize(self):
        """
        Test GPFA orthonormalize function.
        """
        # get sequence spiketrains to be orthonormalized
        seqs_latent, _ = gpfa_core.exact_inference_with_ll(
            self.data, self.params_init
            )
        corth, _ = gpfa_core.orthonormalize(self.params_init, seqs_latent)
        c_orth = linalg.orth(self.params_init['C'])
        # Assert
        self.assertTrue(np.allclose(c_orth, corth))

    def test_logdet(self):
        """
        Test GPFA lodget function.
        """
        np.random.seed(27)
        # generate a positive definite matrix
        matrix = np.random.randn(20, 20)
        matrix = matrix.dot(matrix.T)
        logdet_fast = gpfa_util.logdet(matrix)
        logdet_ground_truth = np.log(np.linalg.det(matrix))
        self.assertAlmostEqual(logdet_fast, logdet_ground_truth)
