import numpy as np
import timeit
import time
import json
from scipy.stats import sem
from gpfa import GPFA, GPFANonInc, GPFAInvPerSymm, GPFAInvPerSymmPar, GPFANonIncPar
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


class GPFAExperiment:
    """Class to manage GPFA simulation parameters and data generation."""

    def __init__(self, rng_seeds=[0, 10, 42, 100], z_dim=3, x_dim=10,
                 tau_f=0.6, sigma_n=0.001, bin_size=0.05,
                 T_per_obs=[500, 500, 500, 500]):
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
    for cls in [GPFA, GPFANonInc, GPFANonIncPar, GPFAInvPerSymm, GPFAInvPerSymmPar]:
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

    avg_em_runtime = np.mean(em_timer.times)
    sem_em_runtime = sem(em_timer.times) if len(em_timer.times) > 1 else 0

    avg_infer_latents_runtime = np.mean(infer_latents_timer.times)
    sem_infer_latents_runtime = sem(infer_latents_timer.times) if \
        len(infer_latents_timer.times) > 1 else 0
    print(
    f"{model_name} average _em runtime: {avg_em_runtime:.4f} ± "
    f"{sem_em_runtime:.4f} seconds\n"
    )
    print(
        f"{model_name} average _infer_latents runtime: {avg_infer_latents_runtime:.4f} ± "
        f"{sem_infer_latents_runtime:.4f} seconds\n"
    )

    # Reset timers for the next model
    em_timer.reset()
    infer_latents_timer.reset()

    return {
        "avg_runtime": avg_runtime,
        "sem_runtime": sem_runtime,
        "avg_em_runtime": avg_em_runtime,
        "sem_em_runtime": sem_em_runtime,
        "avg_infer_latents_runtime": avg_infer_latents_runtime,
        "sem_infer_latents_runtime": sem_infer_latents_runtime,
    }


if __name__ == "__main__":
    # List of varying T_per_obs
    T_per_obs_list = [
        [500, 500, 500, 500], [490, 490, 510, 510], [480, 480, 520, 520],
        [470, 470, 530, 530], [460, 460, 540, 540], [450, 450, 550, 550],
        [440, 440, 560, 560], [430, 430, 570, 570], [420, 420, 580, 580],
        [410, 410, 590, 590], [400, 400, 600, 600]
    ]

    # Number of times to run the experiment per condition
    num_runs = 5

    # Dictionary to store results
    results = {}

    for T_per_obs in T_per_obs_list:
        print(f"\nRunning experiment for T_per_obs={T_per_obs}\n")

        # Initialize experiment with the current T_per_obs
        exp = GPFAExperiment(T_per_obs=T_per_obs)

        # Generate data
        X = exp.generate_data()

        # Instantiate timers
        em_timer = Timer()
        infer_latents_timer = Timer()

        # Patch GPFA methods
        patch_gpfa_methods(em_timer, infer_latents_timer)

        # Run profiling for each model and store results
        results[str(T_per_obs)] = {
            "GPFA_Threaded_BlockInv": profile_model_fit(
                GPFA, "GPFA", num_runs, X, exp.bin_size,
                exp.z_dim, em_timer, infer_latents_timer
                ),
            "GPFA_Threaded_LinalgInv": profile_model_fit(
                GPFANonInc, "GPFANonInc", num_runs, X,
                exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Serial_LinalgInv": profile_model_fit(
                GPFANonIncPar, "GPFANonIncPar", num_runs, X,
                exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Threaded_PersymInv": profile_model_fit(
                GPFAInvPerSymm, "GPFAInvPerSymm", num_runs,
                X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Serial_PersymInv": profile_model_fit(
                GPFAInvPerSymmPar, "GPFAInvPerSymmPar",
                num_runs, X, exp.bin_size, exp.z_dim, 
                em_timer, infer_latents_timer
                ),
        }

    # Save results to a JSON file for further analysis
    with open("gpfa_profiling_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nAll experiments completed! Results saved to gpfa_profiling_results.json\n")
