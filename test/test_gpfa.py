"""Test gpfa.py."""

import unittest
import neo
import numpy as np
from scipy import linalg
import quantities as pq
from sklearn.decomposition import FactorAnalysis

import gpfa
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
        def gen_gamma_spike_train(k, theta, t_max):
            x_gam = []
            for _ in range(int(3 * t_max / (k * theta))):
                x_gam.append(np.random.gamma(k, theta))
            s_gam = np.cumsum(x_gam)
            return s_gam[s_gam < t_max]

        def gen_test_data(rates, dur, shapes=(1, 1, 1, 1)):
            s_gam = gen_gamma_spike_train(shapes[0], 1. / rates[0], dur[0])
            for i in range(1, 4):
                s_i = gen_gamma_spike_train(shapes[i], 1. / rates[i], dur[i])
                s_gam = np.concatenate([s_gam, s_i + np.sum(dur[:i])])
            return s_gam

        # generate data
        self.rates_a = (2, 10, 2, 2)
        self.rates_b = (2, 2, 10, 2)
        self.durs = (2.5, 2.5, 2.5, 2.5)
        np.random.seed(0)
        self.n_trials = 1
        self.data = []
        for _ in range(self.n_trials):
            n_1 = neo.SpikeTrain(gen_test_data(
                self.rates_a, self.durs), units=1 * pq.s,
                t_start=0 * pq.s, t_stop=10 * pq.s
                )
            n_2 = neo.SpikeTrain(gen_test_data(
                self.rates_a, self.durs), units=1 * pq.s,
                t_start=0 * pq.s, t_stop=10 * pq.s
                )
            n_3 = neo.SpikeTrain(gen_test_data(
                self.rates_b, self.durs), units=1 * pq.s,
                t_start=0 * pq.s, t_stop=10 * pq.s
                )
            n_4 = neo.SpikeTrain(gen_test_data(
                self.rates_b, self.durs), units=1 * pq.s,
                t_start=0 * pq.s, t_stop=10 * pq.s
                )
            self.data.append([n_1, n_2, n_3, n_4])

        self.bin_width = 20.0
        self.tau_init = 100.0
        self.eps_init = 1.0E-3
        self.bin_size = 20 * pq.ms
        self.n_iters = 10
        self.x_dim = 2

        # covert generated data to sequence spiketrains
        self.seqs_train = gpfa_util.get_seqs(
                                self.data, self.bin_size, use_sqrt=False
                                )
        # get the number of time steps in the trails
        self.t_all = self.seqs_train['T'][0]
        self.t_half = int(np.ceil(self.t_all / 2.0))

        # Initialize state model parameters
        self.params_init = dict()
        self.params_init['covType'] = 'rbf'
        # GP timescale
        # Assume binWidth is the time step size.
        time_step = (self.bin_width / self.tau_init) ** 2
        self.params_init['gamma'] = time_step * np.ones(self.x_dim)
        # GP noise variance
        self.params_init['eps'] = self.eps_init * np.ones(self.x_dim)

        # Initialize observation model parameters using factor analysis
        y_all = np.hstack(self.seqs_train['y'])
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

        # (y - d)
        dif = np.hstack(self.seqs_train['y']) - \
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
            self.seqs_train, self.params_init
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
        seqs_latent, _ = gpfa.gpfa_core.exact_inference_with_ll(
            self.seqs_train, self.params_init
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
