=======================================
Gaussian-Process Factor Analysis (GPFA)
=======================================

Gaussian-process factor analysis (GPFA) is a dimensionality reduction
method [Yu2009_614]_ that extracts smooth, low-dimensional
latent trajectories from noisy, high-dimensional time series data. GPFA
applies `factor analysis (FA) <https://scikit-learn.org/stable/modules/
generated/sklearn.decomposition.FactorAnalysis.html>`_ to observed data
to reduce the dimensionality and at the same time smoothes the resulting
low-dimensional trajectories by fitting a `Gaussian process (GP)
<https://scikit-learn.org/stable/modules/generated/
sklearn.gaussian_process.kernels.Kernel.
html#sklearn.gaussian_process.kernels.Kernel>`_ model to them.

The generative model underlying GPFA can be described mathematically as
follows:

1. **Latent Variables (Factors):**

   Given a dataset of high-dimensional time series observations represented
   as a matrix :math:`X` with dimensions (``x_dim`` x ``bins``), where ``x_dim``
   is the number of observed dimensions (features) and ``bins`` is the number of
   time points.

   The goal of GPFA is to infer the underlying latent trajectories or factors
   that govern the observed data. The latent variable matrix is denoted
   as :math:`Z` with dimensions (``z_dim`` x ``bins``), where ``z_dim`` is the
   number of latent dimensions (factors).

2. **Latent Variable Prior Distribution:**

   GPFA assumes that the latent trajectories ``Z`` follow a Gaussian Process
   prior. A Gaussian Process is a collection of random variables, any finite
   number of which have a joint Gaussian distribution [RW2006]_. The prior
   over latent trajectories is defined as:

    :math:`Z \sim \mathcal{GP}(0, K)`, where:

    - :math:`0` is the mean function (zero-mean assumption).
    - :math:`K` is the covariance matrix that characterizes
      the smoothness and correlations between different latent
      dimensions.

3. **Emission Model:**

   GPFA assumes a linear relationship between the latent trajectories :math:`Z`
   and the observed data :math:`X`. The emission model is represented as:

    :math:`X = CZ + d + Guass(0, R)`, where:

     - :math:`C` is the loading matrix with dimensions (``x_dim`` x ``K``),
       representing the linear mapping from latent trajectories to observed data.
     - :math:`d` is the bias term with dimensions (``x_dim``, 1), accounting
       for any mean shift in the observed data.
     - :math:`Gauss(0, R)` is the noise term with dimensions
       (``x_dim`` x ``bins``), representing the observation noise and assumed to
       be Gaussian with zero mean.

4. **Observation Model:**

   The observation model assumes that the observed data :math:`X` follows a
   multivariate Gaussian distribution conditioned on the latent trajectories
   ``Z``:

    :math:`X|Z \sim \mathcal{N}(CZ + d, R)`, where:

    - :math:`CZ + d` is the mean of the Gaussian distribution.
    - :math:`R` is the covariance matrix capturing the uncertainty in the
      observed data.

   The observation model is also known as the likelihood function.

5. **Model Parameters:**

   The primary parameters requiring estimation in the original implementation
   of GPFA are the loading matrix :math:`C`, the bias term :math:`d`, the
   covariance matrix :math:`R`, and the characteristic timescales :math:`\theta`.
   In the current implementation, supplementary parameters might encompass the
   signal variances along with the Gaussian Process (GP) noise variance,
   contingent upon the user's specification of the ``gp_kernel``. See notes on
   ``gp_kernel``.

6. **Inference:**

   Inference in GPFA employs the Expectation-Maximization (EM) algorithm, a
   cyclic optimization technique for parameter estimation in probabilistic
   models with latent trajectories. It improves model parameters by alternately
   estimating latent variable posteriors (E-step) and enhancing parameters
   to maximize the expected complete data log-likelihood (M-step). This
   iterative process leads to convergence, yielding parameter estimates
   capturing underlying data structure.

Notes (TODO)
============
.. These are some of the things worthy of note, but I think they are better
.. fitted in the GPFA class documentation (we can discuss this).
.. - Differences between the original implementation and the current one:
  
..   - Using :func:`use_cut_trials`, pros and cons
..   - block inverse computation
..   - GP Kernel
..   - Variance explained

.. _examples:

Examples (TODO)
===============
.. Discuss appropriate examples

>>> import numpy as np
>>> from gpfa import GPFA
>>> from sklearn.gaussian_process.kernels import RBF, WhiteKernel
>>> from sklearn.gaussian_process.kernels import ConstantKernel

>>> # set random parameters
>>> seed = [0, 8, 10]
>>> z_dim = 3
>>> units = 10
>>> tau_f = 0.1
>>> sigma_n = 0.001
>>> sigma_f = 1 - sigma_n
>>> bin_size = 0.02  # [s]
>>> num_trials = 3
>>> n_timesteps = 500
>>> kernel = ConstantKernel(
...                    sigma_f, constant_value_bounds='fixed'
...                    ) * RBF(length_scale=tau_f) + ConstantKernel(
...                    sigma_n, constant_value_bounds='fixed'
...                    ) * WhiteKernel(
...                        noise_level=1, noise_level_bounds='fixed'
...                    )
>>> tsdt = np.arange(0, n_timesteps) * bin_size
>>> mu = np.zeros(tsdt.shape)
>>> cov = kernel(tsdt[:, np.newaxis])
>>> C = np.random.uniform(0, 2, (units, z_dim))     # loading matrix
>>> obs_noise = np.random.uniform(0.2, 0.75, units) # rand noise parameters
>>> X = []
>>> for n in range(num_trials):
>>>     np.random.seed(seed[n])
>>>     # Draw three latent state samples from a Gaussian process
>>>     # using the above cov
>>>     Z = np.random.multivariate_normal(mu.ravel(), cov, z_dim)
>>>     # observations have Gaussian noise
>>>     x = C @ Z + np.random.normal(0, obs_noise, (n_timesteps, units)).T
>>>     X.append(x)
>>> gpfa = GPFA(bin_size=bin_size, z_dim=z_dim)
>>> gpfa.fit(X)
Initializing parameters using factor analysis...
Fitting GPFA model...
>>> results, _ = gpfa.predict(returned_data=['pZ_mu', 'pZ_mu_orth'])
>>> pZ_mu_orth = results['pZ_mu_orth']
>>> pZ_mu = results['pZ_mu']
>>> gpfa.variance_explained()
(0.93590..., array([0.76541..., 0.10446..., 0.066033...]))

>>> # GPFA on synthetic spike data
>>> import numpy as np
>>> from gpfa import GPFA
>>> from gpfa.preprocessing import EventTimesToCounts
>>> from sklearn.preprocessing import FunctionTransformer
>>> seed = [0, 8, 10, 42, 60]
>>> rate = 50
>>> units = 10
>>> durations = [500, 550, 600, 650, 700]  # [ms]
>>> num_trials = len(durations)
>>> X = np.zeros(num_trials, object)
>>> for i in range(num_trials):
>>>     np.random.seed(seed[i])
>>>     Data[i] = np.random.poisson(rate, (units, durations[i]))
>>> event_times_to_counts = EventTimesToCounts(extrapolate_last_bin=True)
>>> binned_spiketrians = [
...    event_times_to_counts.transform(x_i) for x_i in X
...    ]
>>> fun_trans = FunctionTransformer(np.sqrt)
>>> sqrt_spike_trains = [
...    fun_trans.transform(x_i) for x_i in binned_spiketrians
...    ]
>>> z_dim = 3
>>> bin_size = 0.02  # [s]
>>> gpfa = GPFA(bin_size=bin_size, z_dim=z_dim, em_max_iters=2)
>>> gpfa.fit(X)
Initializing parameters using factor analysis...
Fitting GPFA model...
>>> results, _ = gpfa.predict(returned_data=['pZ_mu','pZ_mu_orth'])
>>> pZ_mu_orth = results['pZ_mu_orth']
>>> pZ_mu = results['pZ_mu']
>>> gpfa.variance_explained()
(0.98518..., array([9.85162126e-01, 1.32456401e-05, 7.66699002e-06]))

Original code
-------------
The code was ported from the MATLAB code based on Byron Yu's implementation.
The original MATLAB code is available at Byron Yu's website:
https://users.ece.cmu.edu/~byronyu/software.shtml

References
----------
.. [Yu2009_614] `Yu, Byron M and Cunningham, John P and Santhanam, Gopal and
    Ryu, Stephen and Shenoy, Krishna V and Sahani, Maneesh
    "Gaussian-process factor analysis for low-dimensional single-trial
    analysis of neural population activity"
    In Journal of Neurophysiology, Vol. 102, Issue 1. pp. 614-635.
    <https://doi.org/10.1152/jn.90941.2008>`_

.. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
   "Gaussian Processes for Machine Learning",
   MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

.. _main_page:
.. toctree::
    :maxdepth: 2

    gpfa_class
    preprocessing_class
    installation
    contribute
    acknowledgments
    authors
    citation
