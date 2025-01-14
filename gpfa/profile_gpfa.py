import numpy as np
from gpfa import GPFA
from gpfa_without_block_inversion import GpfaWithoutBlockInv
from gpfa_inv_persymm import GpfaInvPerSymmetric
from line_profiler import LineProfiler

# Simulation parameters
rng_seeds = [0, 10, 42]
z_dim = 2
x_dim = 10
tau_f = 0.6
sigma_n = 0.001
sigma_f = 1 - sigma_n
bin_size = 0.05  # [s]
num_obs = len(rng_seeds)
T_per_obs = [350, 400, 450]

# Define kernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor

kernel = ConstantKernel(sigma_f, constant_value_bounds='fixed') * RBF(length_scale=tau_f) + \
        ConstantKernel(sigma_n, constant_value_bounds='fixed') * WhiteKernel(noise_level=1, noise_level_bounds='fixed')

# Generate data
np.random.seed(2)
C = np.random.uniform(0, 2, (x_dim, z_dim))
sqrtR = np.random.uniform(0, 0.5, x_dim)

X = []
Z = []
for n in range(num_obs):
    np.random.seed(rng_seeds[n])
    tsdt = np.arange(0, T_per_obs[n]) * bin_size
    gp_model = GaussianProcessRegressor(kernel=kernel)
    z = gp_model.sample_y(tsdt[:, np.newaxis], n_samples=z_dim, random_state=rng_seeds[n]).T
    Z.append(z)
    x = C @ z + np.random.normal(0, sqrtR[:, np.newaxis], (x_dim, T_per_obs[n]))
    X.append(x)

# Initialize models
gpfa = GPFA(bin_size=bin_size, z_dim=z_dim, em_tol=1e-3, verbose=False)
gpfa_without_block_inv = GpfaWithoutBlockInv(bin_size=bin_size, z_dim=z_dim, em_tol=1e-3, verbose=False)
gpfa_inv_persymm = GpfaInvPerSymmetric(bin_size=bin_size, z_dim=z_dim, em_tol=1e-3, verbose=False)

# Create a LineProfiler instance
profiler = LineProfiler()

# Profile the `fit` methods
profiler.add_function(gpfa.fit)
profiler.add_function(gpfa_without_block_inv.fit)
profiler.add_function(gpfa_inv_persymm.fit)

# Run the profiling
profiler.enable()
gpfa.fit(X)
gpfa_without_block_inv.fit(X)
gpfa_inv_persymm.fit(X)
profiler.disable()

# Print the profiling results
profiler.print_stats()
