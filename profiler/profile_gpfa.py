import numpy as np
import timeit
import time
from scipy.stats import sem
from gpfa import GPFA, GPFANonInc, GPFAInvPerSymm
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


class GPFAExperiment:
    """Class to manage GPFA simulation parameters and data generation."""

    def __init__(self, rng_seeds=[0, 10, 42], z_dim=2, x_dim=10,
                 tau_f=0.6, sigma_n=0.001, bin_size=0.05,
                 T_per_obs=[400, 400, 400]):
        self.rng_seeds = rng_seeds
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.tau_f = tau_f
        self.sigma_n = sigma_n
        self.sigma_f = 1 - sigma_n
        self.bin_size = bin_size
        self.T_per_obs = T_per_obs
        
        # Validate parameters
        if len(self.T_per_obs) != len(self.rng_seeds):
            raise ValueError("T_per_obs must match the number of rng_seeds.")

        # Kernel setup
        self.kernel = (
            ConstantKernel(self.sigma_f, constant_value_bounds="fixed")
            * RBF(length_scale=self.tau_f)
            + ConstantKernel(self.sigma_n, constant_value_bounds="fixed")
            * WhiteKernel(noise_level=1, noise_level_bounds="fixed")
        )

    def generate_data(self):
        """Generate simulated data based on the current parameters."""
        np.random.seed(2)
        C = np.random.uniform(0, 2, (self.x_dim, self.z_dim))
        sqrtR = np.random.uniform(0, 0.5, self.x_dim)

        X = []
        for n, seed in enumerate(self.rng_seeds):
            np.random.seed(seed)
            tsdt = np.arange(0, self.T_per_obs[n]) * self.bin_size
            gp_model = GaussianProcessRegressor(kernel=self.kernel)
            z = gp_model.sample_y(
                tsdt[:, np.newaxis], n_samples=self.z_dim, random_state=seed
                ).T
            x = C @ z + np.random.normal(
                0, sqrtR[:, np.newaxis], (self.x_dim, self.T_per_obs[n])
                )
            X.append(x)
        return X


# Timer class for profiling
class Timer:
    """Utility class to measure and store method runtimes."""
    def __init__(self):
        self.times = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            self.times.append(elapsed_time)
            return result
        return wrapper

    def reset(self):
        """Reset the stored runtimes."""
        self.times = []


# Patch methods for profiling
def patch_gpfa_methods(em_timer, infer_latents_timer):
    """Patch GPFA and its variants to include profiling."""
    for cls in [GPFA, GPFANonInc, GPFAInvPerSymm]:
        cls._em = em_timer(cls._em)
        cls._infer_latents = infer_latents_timer(cls._infer_latents)


# Profiling functions
def profile_model_fit(
        model_cls, model_name, num_runs, X, bin_size, z_dim, em_timer, infer_latents_timer
    ):
    """Profile the fit, _em and _infer_latents methods of a given GPFA model class."""
    
    def model_fit():
        model = model_cls(bin_size=bin_size, z_dim=z_dim, em_tol=1e-3, verbose=2)
        model.fit(X)

    # Collect individual runtimes
    runtimes = [timeit.timeit(model_fit, number=1) for _ in range(num_runs)]

    # Compute mean and SEM
    avg_runtime = np.mean(runtimes)
    sem_runtime = sem(runtimes) if len(runtimes) > 1 else 0

    print(f"\n{model_name} fit average runtime: {avg_runtime:.4f} ± {sem_runtime:.4f} seconds\n")

    # Compute mean and SEM for _em method
    avg_em_runtime = np.mean(em_timer.times)
    sem_em_runtime = sem(em_timer.times) if len(em_timer.times) > 1 else 0

    # Compute mean and SEM for _infer_latents method
    avg_infer_latents_runtime = np.mean(infer_latents_timer.times)
    sem_infer_latents_runtime = sem(infer_latents_timer.times) if len(infer_latents_timer.times) > 1 else 0

    print(f"{model_name} average _em runtime: {avg_em_runtime:.4f} ± {sem_em_runtime:.4f} seconds\n")
    print(f"{model_name} average _infer_latents runtime: {avg_infer_latents_runtime:.4f} ± {sem_infer_latents_runtime:.4f} seconds\n")

    # Reset timers for the next model
    em_timer.reset()
    infer_latents_timer.reset()


if __name__ == "__main__":
    # User-specified parameters
    exp = GPFAExperiment()

    # Generate data
    X = exp.generate_data()

    # Instantiate timers
    em_timer = Timer()
    infer_latents_timer = Timer()

    # Patch GPFA methods
    patch_gpfa_methods(em_timer, infer_latents_timer)

    # Profile each model
    num_runs = 5
    profile_model_fit(
        GPFA, "GPFA", num_runs, X, exp.bin_size,
        exp.z_dim, em_timer, infer_latents_timer
        )
    profile_model_fit(
        GPFANonInc, "GPFANonInc", num_runs, X, exp.bin_size,
        exp.z_dim, em_timer, infer_latents_timer
        )
    profile_model_fit(
        GPFAInvPerSymm, "GPFAInvPerSymm", num_runs, X, exp.bin_size,
        exp.z_dim, em_timer, infer_latents_timer)
