import numpy as np
import logging
from benchmark import Benchmark
from solvers import (
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Prints logs to the console
        logging.FileHandler(
            "benchmark-lindblad-solver.log", mode="w"
        ),  # Saves logs to a file
    ],
)

if __name__ == "__main__":
    # Simulation parameters
    time_span = 10  # Total time in multiples of tau
    samples_per_tau = 100  # Number of samples per tau
    max_atoms = 7  # Maximum number of atoms for scalability tests
    coupling = 1.0  # Coupling strength
    decay_rate = 2 * np.pi * 5.22e6  # Decay rate for Cs-133 D2 transition (rad/s)

    # List of solver classes to benchmark
    solvers = [
        QuTiPLindbladSolver,
        SciPyLindbladSolver,
        RungeKuttaLindbladSolver,
        BackwardEulerLindbladSolver,
    ]

    # Initialize Benchmark instance
    benchmark = Benchmark()

    # Run scalability tests for each solver
    for solver_class in solvers:
        benchmark.test_scalability(
            solver_class, max_atoms, time_span, samples_per_tau, coupling, decay_rate
        )

    # Display the scalability results
    benchmark.display_results()
