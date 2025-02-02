import pytest
from src.benchmark import Benchmark
from src.solvers import QuTiPLindbladSolver


def test_benchmark_coherent():
    time_span = 10  # Total time in multiples of decay time
    samples_per_decay_time = 100  # Number of samples per decay time
    n_atoms = 2  # number of atoms for the test
    decay_rate = 1e-6  # Decay rate
    coupling = 1

    benchmark = Benchmark(
        n_atoms, time_span, samples_per_decay_time, coupling, decay_rate
    )

    solver = QuTiPLindbladSolver(
        time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    )
    results = benchmark.run("Test Solver", solver)
    assert "runtime" in results, "Benchmark did not return runtime."
    assert "memory_usage" in results, "Benchmark did not return memory usage."


def test_benchmark_dissipative():
    time_span = 10  # Total time in multiples of decay time
    samples_per_decay_time = 100  # Number of samples per decay time
    n_atoms = 2  # number of atoms for the test
    decay_rate = 1e6
    coupling = 1

    benchmark = Benchmark(
        n_atoms, time_span, samples_per_decay_time, coupling, decay_rate
    )

    solver = QuTiPLindbladSolver(
        time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    )
    results = benchmark.run("Test Solver", solver)
    assert "runtime" in results, "Benchmark did not return runtime."
    assert "memory_usage" in results, "Benchmark did not return memory usage."
