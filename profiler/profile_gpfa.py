# ...
# Moving this here to prevent from showing up on sphinx pages
# Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
# Copyright 2014-2020 by the Elephant team.
# Modified BSD, see LICENSE.txt for details.
# ...

import timeit
import time
import json
import numpy as np
from scipy.stats import sem
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from gpfa import (
    GPFA,
    GPFASerialBlockInv,
    GPFASerialLinalgInv,
    GPFASerialPersymInv,
    GPFAThreadedLinalgInv,
    GPFAThreadedPersymInv
    )


class GPFAExperiment:
    """Class to manage GPFA simulation parameters and data generation."""

    def __init__(self, rng_seeds=[0, 10, 42, 100], z_dim=4, x_dim=10,
                 tau_f=0.1, sigma_n=0.001, bin_size=0.06,
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
            raise ValueError(f"T_per_obs must match the number of rng_seeds.")

        # Kernel setup
        self.kernel = (
            ConstantKernel(self.sigma_f, constant_value_bounds="fixed")
            * RBF(length_scale=max(self.tau_f, 1e-2))
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
    for cls in [GPFA,  GPFASerialBlockInv,
                GPFASerialLinalgInv, GPFASerialPersymInv,
                GPFAThreadedLinalgInv, GPFAThreadedPersymInv]:
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
        [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
        [495, 495, 495, 495, 495, 505, 505, 505, 505, 505],
        [491, 491, 491, 491, 501, 501, 501, 511, 511, 511],
        [487, 487, 487, 497, 497, 497, 507, 507, 517, 517],
        [480, 480, 490, 490, 500, 500, 510, 510, 520, 520],
        [479, 479, 489, 489, 499, 499, 509, 509, 519, 529],
        [476, 476, 486, 486, 496, 496, 506, 516, 526, 536],
        [471, 471, 481, 481, 491, 501, 511, 521, 531, 541],
        [464, 464, 474, 484, 494, 504, 514, 524, 534, 544],
        [455, 465, 475, 485, 495, 505, 515, 525, 535, 545],
    ]
    # Number of times to run the experiment per condition
    num_runs = 5

    # Dictionary to store results
    results = {}

    rng_seeds = [0, 10, 42, 100, 0, 10, 42, 100, 0, 10]

    for T_per_obs in T_per_obs_list:

        print(f"\nRunning experiment for T_per_obs={T_per_obs}\n")

        # Initialize experiment with the current T_per_obs
        exp = GPFAExperiment(rng_seeds=rng_seeds, T_per_obs=T_per_obs)

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
                GPFAThreadedLinalgInv, "GPFA_Threaded_LinalgInv",
                num_runs, X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Threaded_PersymInv": profile_model_fit(
                GPFAThreadedPersymInv, "GPFA_Threaded_PersymInv",
                num_runs, X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Serial_BlockInv": profile_model_fit(
                GPFASerialBlockInv, "GPFA_Serial_BlockInv",
                num_runs, X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Serial_LinalgInv": profile_model_fit(
                GPFASerialLinalgInv, "GPFA_Serial_LinalgInv",
                num_runs, X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                ),
            "GPFA_Serial_PersymInv": profile_model_fit(
                GPFASerialPersymInv, "GPFA_Serial_PersymInv",
                num_runs, X, exp.bin_size, exp.z_dim, em_timer,
                infer_latents_timer
                )
        }

    with open("gpfa_profiling_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nAll experiments completed! Results saved to gpfa_profiling_results.json\n")
