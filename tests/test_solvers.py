import pytest
from src.solvers import (
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
)


@pytest.fixture
def solver_params():
    """
    Fixture for providing parameters for the Lindblad solvers.

    The parameters are:
        - t_list: A list of time points for solving the equation.
        - n_atoms: The number of atoms in the system.
        - coupling: The coupling strength between the atoms.
        - decay_rate: The spontaneous emission rate.

    Returns:
        A dictionary with the above parameters.
    """
    return {
        "t_list": [0, 1, 2],
        "n_atoms": 2,
        "coupling": 1.0,
        "decay_rate": 0.1,
    }


def test_qutip_solver(solver_params):
    """
    Test that the QuTiP solver produces a result.

    The test checks that the QuTiP solver produces a result, which is a QuTiP
    Result object. The test fails if the result is None.
    """
    solver = QuTiPLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "QuTiP solver failed to produce a result."


def test_scipy_solver(solver_params):
    """
    Test that the SciPy solver successfully solves the Lindblad equation.

    The test verifies that the SciPy solver produces a successful result,
    indicated by the `success` attribute of the result object. The test
    fails if the solver does not successfully solve the equation.
    """

    solver = SciPyLindbladSolver(**solver_params)
    result = solver.solve()
    assert result.success, "SciPy solver failed to solve the equation."


def test_runge_kutta_solver(solver_params):
    """
    Test that the Runge-Kutta solver produces a result.

    The test checks that the Runge-Kutta solver produces a result, which is a
    NumPy array. The test fails if the result is None.
    """
    solver = RungeKuttaLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "Runge-Kutta solver failed to produce a result."


def test_backward_euler_solver(solver_params):
    """
    Test that the Backward Euler solver produces a result.

    The test checks that the Backward Euler solver produces a result, which is a
    NumPy array. The test fails if the result is None.
    """
    solver = BackwardEulerLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "Backward Euler solver failed to produce a result."
