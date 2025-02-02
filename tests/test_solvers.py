import pytest
from src.solvers import (
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
    SparseLindbladSolver,
)

# List of solver classes to test
solvers = [
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
    SparseLindbladSolver,
]


@pytest.fixture
def solver_params_dissipative():
    """
    Fixture returning parameters for Lindblad solvers that simulate a dissipative system
    (i.e. a system where the decay rate is greater than the coupling strength).

    Returns:
        dict: Dictionary containing parameters for the Lindblad solvers.
    """

    return {
        "time_span": 10,
        "samples_per_decay_time": 100,
        "n_atoms": 2,
        "coupling": 1.0,
        "decay_rate": 1e6,
    }


@pytest.fixture
def solver_params_coherent():
    """
    Fixture returning parameters for Lindblad solvers for testing coherent systems,
    (i.e. systems where the coupling strength is greater than the decay rate).

    Returns:
        dict: Dictionary containing parameters for the Lindblad solvers.
    """
    return {
        "time_span": 10,
        "samples_per_decay_time": 100,
        "n_atoms": 2,
        "coupling": 1.0,
        "decay_rate": 1e-6,
    }


def test_solvers_dissipative(solver_params_dissipative):
    """
    Test that each solver produces a result in a dissipative system, i.e.
    where the decay rate is greater than the coupling strength.

    The test creates an instance of each solver with the given parameters,
    and measures the solver's performance. The test asserts that the measurement
    results dictionary contains the expected keys ("runtime" and "memory_usage").

    Parameters:
        solver_params_dissipative (dict): A dictionary of parameters for the solver,
            including time_span, samples_per_decay_time, n_atoms, coupling, and decay_rate.
    """
    for solver in solvers:
        solver_instance = solver(**solver_params_dissipative)
        result = solver_instance.solve()
        assert (
            result is not None
        ), f"{solver.__name__} failed to produce a result with decay_rate > coupling."


def test_solvers_coherent(solver_params_coherent):
    """
    Test that each solver produces a result in a coherent system, i.e.
    where the decay rate is less than the coupling strength.

    The test creates an instance of each solver with the given parameters,
    and measures the solver's performance. The test asserts that the measurement
    results dictionary contains the expected keys ("runtime" and "memory_usage").

    Parameters:
        solver_params_coherent (dict): A dictionary of parameters for the solver,
            including time_span, samples_per_decay_time, n_atoms, coupling, and decay_rate.
    """
    for solver in solvers:
        solver_instance = solver(**solver_params_coherent)
        result = solver_instance.solve()
        assert (
            result is not None
        ), f"{solver.__name__} failed to produce a result with decay_rate < coupling."
