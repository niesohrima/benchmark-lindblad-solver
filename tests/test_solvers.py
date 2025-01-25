import pytest
from src.solvers import (
    QuTiPLindbladSolver,
    SciPyLindbladSolver,
    RungeKuttaLindbladSolver,
    BackwardEulerLindbladSolver,
)

@pytest.fixture
def solver_params():
    return {
        "t_list": [0, 1, 2],
        "n_atoms": 2,
        "coupling": 1.0,
        "decay_rate": 0.1,
    }

def test_qutip_solver(solver_params):
    solver = QuTiPLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "QuTiP solver failed to produce a result."

def test_scipy_solver(solver_params):
    solver = SciPyLindbladSolver(**solver_params)
    result = solver.solve()
    assert result.success, "SciPy solver failed to solve the equation."

def test_runge_kutta_solver(solver_params):
    solver = RungeKuttaLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "Runge-Kutta solver failed to produce a result."

def test_backward_euler_solver(solver_params):
    solver = BackwardEulerLindbladSolver(**solver_params)
    result = solver.solve()
    assert result is not None, "Backward Euler solver failed to produce a result."