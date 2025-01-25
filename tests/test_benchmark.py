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
    t_list = [0, 1, 2]
    solver = QuTiPLindbladSolver(t_list, 2, 1.0, 0.1)
    results = benchmark.run("Test Solver", solver)
    assert "runtime" in results, "Benchmark did not return runtime."
    assert "memory_usage" in results, "Benchmark did not return memory usage."
