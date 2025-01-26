from datetime import datetime
import numpy as np
import logging
from benchmark import Benchmark
from solvers import (
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
)

# Generate a timestamp
benchmark_time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Prints logs to the console
        logging.FileHandler(
            f"benchmark-{benchmark_time_stamp}.log", mode="w"
        ),  # Saves logs to a file
    ],
)

if __name__ == "__main__":
    # Simulation parameters
    time_span = 10  # Total time in multiples of tau
    samples_per_tau = 100  # Number of samples per tau
    max_atoms = 6  # Maximum number of atoms for scalability tests
    coupling = 1.0  # Coupling strength
    decay_rate = 2 * np.pi * 5.22e6  # Decay rate for Cs-133 D2 transition (rad/s)

    if max_atoms < 2:
        raise ValueError("max_atoms must be at least 2.")

    # List of solver classes to benchmark
    solvers = [
        QuTiPLindbladSolver,
        SciPyLindbladSolver,
        RungeKuttaLindbladSolver,
        BackwardEulerLindbladSolver,
    ]

    # Initialize Benchmark instance
    benchmark = Benchmark(max_atoms, time_span, samples_per_tau, coupling, decay_rate)

    # Run scalability tests for each solver
    for solver_class in solvers:
        benchmark.test_scalability(solver_class)

    # Display the scalability results
    benchmark.display_results(output_file=f"benchmark-plots-{benchmark_time_stamp}.png")
