import pytest
from src.benchmark import Benchmark
from src.solvers import QuTiPLindbladSolver


def test_benchmark():
    """
    Test that the Benchmark class can be used to measure the performance of a solver.

    The test creates a Benchmark instance, defines a solver, and uses the Benchmark
    to measure the performance of the solver. The test asserts that the measurement
    results dictionary contains the expected keys ("runtime" and "memory_usage").

    """
    benchmark = Benchmark()
    time_span = 10  # Total time in multiples of tau
    samples_per_tau = 100  # Number of samples per tau
    n_atoms = 2  # number of atoms for the test
    coupling = 1.0  # Coupling strength
    decay_rate = (
        2 * 3.1415 * 5.22e6
    )  # approx. decay rate for Cs-133 D2 transition (rad/s)

    solver = QuTiPLindbladSolver(
        time_span, samples_per_tau, n_atoms, coupling, decay_rate
    )
    results = benchmark.run("Test Solver", solver)
    assert "runtime" in results, "Benchmark did not return runtime."
    assert "memory_usage" in results, "Benchmark did not return memory usage."
