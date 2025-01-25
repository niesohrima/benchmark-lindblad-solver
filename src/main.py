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
    t_list = np.linspace(0, 10, 100)  # Time points
    max_atoms = 7  # Maximum number of atoms for scalability tests
    coupling = 1.0  # Coupling strength
    decay_rate = 0.1  # Spontaneous decay rate

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
            solver_class, max_atoms, t_list, coupling, decay_rate
        )

    # Display the scalability results
    benchmark.display_results()
