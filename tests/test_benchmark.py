import pytest
from src.benchmark import Benchmark
from src.solvers import QuTiPLindbladSolver


def test_benchmark_coherent():
    """
    Test the coherent benchmark for a Lindblad solver.

    This test evaluates the performance of the QuTiP-based Lindblad solver by
    running a benchmark with a small number of atoms. It verifies that the
    benchmark returns both runtime and memory usage metrics.

    The benchmark is executed using:
    - Time span: 10 decay times
    - Samples per decay time: 100
    - Number of atoms: 2
    - Decay rate: Approximately the decay rate for Cs-133 D2 transition
    - Coupling strength: Twice the decay rate

    Asserts:
        - The benchmark results contain a "runtime" entry.
        - The benchmark results contain a "memory_usage" entry.
    """

    time_span = 10  # Total time in multiples of decay time
    samples_per_decay_time = 100  # Number of samples per decay time
    n_atoms = 2  # number of atoms for the test
    decay_rate = (
        2 * 3.1415 * 5.22e6
    )  # approx. decay rate for Cs-133 D2 transition (rad/s)
    coupling = 2 * decay_rate  # Coupling strength

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
    """
    Test that the Benchmark class can be used to measure the performance of a solver
    in dissipative systems.

    The test creates a Benchmark instance, defines a solver, and uses the Benchmark
    to measure the performance of the solver. The test asserts that the measurement
    results dictionary contains the expected keys ("runtime" and "memory_usage").

    """

    time_span = 10  # Total time in multiples of decay time
    samples_per_decay_time = 100  # Number of samples per decay time
    n_atoms = 2  # number of atoms for the test
    decay_rate = (
        2 * 3.1415 * 5.22e6
    )  # approx. decay rate for Cs-133 D2 transition (rad/s)
    coupling = 0.5 * decay_rate  # Coupling strength

    benchmark = Benchmark(
        n_atoms, time_span, samples_per_decay_time, coupling, decay_rate
    )

    solver = QuTiPLindbladSolver(
        time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    )
    results = benchmark.run("Test Solver", solver)
    assert "runtime" in results, "Benchmark did not return runtime."
    assert "memory_usage" in results, "Benchmark did not return memory usage."
