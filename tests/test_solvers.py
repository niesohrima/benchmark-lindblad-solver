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
    Fixture providing default parameters for solver tests.

    The fixture returns a dictionary containing the default parameters for the
    solvers. The parameters are used in the tests to create a solver instance.

    The default parameters are:

    * time_span: 10
    * samples_per_tau: 100
    * n_atoms: 2
    * coupling: 1.0
    * decay_rate: 2 * 3.1415 * 5.22e6

    The fixture is used in the tests to create a solver instance with the default
    parameters. The fixture is invoked by adding the parameter name to the test
    function.

    Example:

    def test_solver(solver_params):
        solver = Solver(**solver_params)
        result = solver.solve()
        assert result is not None, "Solver failed to produce a result."

    In this example, the test function `test_solver` is invoked with the parameter
    `solver_params`, which is the dictionary returned by this fixture.
    """
    return {
        "time_span": 10,
        "samples_per_tau": 100,
        "n_atoms": 2,
        "coupling": 1.0,
        "decay_rate": 2 * 3.1415 * 5.22e6,
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
