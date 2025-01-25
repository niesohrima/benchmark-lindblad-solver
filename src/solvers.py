import numpy as np
from qutip import *
from scipy.integrate import solve_ivp
import logging


# Base Solver Class
class LindbladSolver:
    def __init__(self, t_list, n_atoms):
        self.t_list = t_list
        self.n_atoms = n_atoms

    def solve(self):
        raise NotImplementedError("Subclasses must implement the solve method.")


# QuTiP Solver
class QuTiPLindbladSolver(LindbladSolver):
    def __init__(self, t_list, n_atoms, coupling, decay_rate):
        super().__init__(t_list, n_atoms)
        self.coupling = coupling
        self.decay_rate = decay_rate

    def _define_hamiltonian(self):
        logging.info(
            f"Defining Hamiltonian for QuTiP solver with {self.n_atoms} atoms."
        )
        H = sum(
            self.coupling
            * (
                tensor([create(2) if i == j else qeye(2) for i in range(self.n_atoms)])
                * tensor(
                    [destroy(2) if i == k else qeye(2) for i in range(self.n_atoms)]
                )
                + tensor(
                    [destroy(2) if i == j else qeye(2) for i in range(self.n_atoms)]
                )
                * tensor(
                    [create(2) if i == k else qeye(2) for i in range(self.n_atoms)]
                )
            )
            for j in range(self.n_atoms)
            for k in range(j + 1, self.n_atoms)
        )
        return H

    def _define_dissipators(self):
        logging.info(
            f"Defining dissipators for QuTiP solver with {self.n_atoms} atoms."
        )
        c_ops = [
            np.sqrt(self.decay_rate)
            * tensor([destroy(2) if i == j else qeye(2) for i in range(self.n_atoms)])
            for j in range(self.n_atoms)
        ]
        return c_ops

    def _define_initial_state(self):
        logging.info(
            f"Defining initial state for QuTiP solver with {self.n_atoms} atoms."
        )
        psi0 = sum(
            tensor(
                [basis(2, 1) if i == j else basis(2, 0) for i in range(self.n_atoms)]
            )
            for j in range(self.n_atoms)
        ).unit()
        return psi0 * psi0.dag()

    def solve(self):
        logging.info(
            f"Solving Lindblad equation with QuTiP solver with {self.n_atoms} atoms."
        )
        H = self._define_hamiltonian()
        c_ops = self._define_dissipators()
        rho0 = self._define_initial_state()
        return mesolve(H, rho0, self.t_list, c_ops)


# SciPy Solver
class SciPyLindbladSolver(LindbladSolver):
    def __init__(self, t_list, n_atoms, coupling, decay_rate):
        super().__init__(t_list, n_atoms)
        self.coupling = coupling
        self.decay_rate = decay_rate

    def _define_hamiltonian(self):
        logging.info(
            f"Defining Hamiltonian for SciPy solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        H = np.zeros((size, size), dtype=complex)

        for j in range(self.n_atoms):
            for k in range(j + 1, self.n_atoms):
                op_j = [qeye(2) if i != j else sigmap() for i in range(self.n_atoms)]
                op_k = [qeye(2) if i != k else sigmam() for i in range(self.n_atoms)]
                H += self.coupling * tensor(op_j).full() @ tensor(op_k).full()
        return H

    def _define_lindblad_operators(self):
        logging.info(
            f"Defining Lindblad operators for SciPy solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        L_ops = []

        for i in range(self.n_atoms):
            L = np.zeros((size, size), dtype=complex)
            for j in range(size // 2):
                L[j, j + size // 2] = np.sqrt(self.decay_rate)
            L_ops.append(L)
        return L_ops

    def _lindblad_rhs(self, t, rho_vec, H, L_ops):
        logging.debug(
            f"Evaluating RHS of Lindblad equation at time {t} with {self.n_atoms} atoms."
        )
        rho = rho_vec.reshape(H.shape)
        commutator = -1j * (H @ rho - rho @ H)
        dissipator = sum(
            L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            for L in L_ops
        )
        return (commutator + dissipator).ravel()

    def solve(self):
        logging.info(
            f"Solving Lindblad equation with SciPy solver with {self.n_atoms} atoms."
        )
        H = self._define_hamiltonian()
        L_ops = self._define_lindblad_operators()
        rho0 = np.zeros((2**self.n_atoms, 2**self.n_atoms))
        rho0[0, 0] = 1.0
        rho0_vec = rho0.ravel()
        return solve_ivp(
            self._lindblad_rhs,
            [self.t_list[0], self.t_list[-1]],
            rho0_vec,
            t_eval=self.t_list,
            args=(H, L_ops),
        )


# Runge-Kutta Solver
class RungeKuttaLindbladSolver(LindbladSolver):
    def __init__(self, t_list, n_atoms, coupling, decay_rate):
        super().__init__(t_list, n_atoms)
        self.coupling = coupling
        self.decay_rate = decay_rate

    def _define_hamiltonian(self):
        logging.info(
            f"Defining Hamiltonian for Runge-Kutta solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        H = np.zeros((size, size), dtype=complex)

        for j in range(self.n_atoms):
            for k in range(j + 1, self.n_atoms):
                op_j = [qeye(2) if i != j else sigmap() for i in range(self.n_atoms)]
                op_k = [qeye(2) if i != k else sigmam() for i in range(self.n_atoms)]
                H += self.coupling * tensor(op_j).full() @ tensor(op_k).full()
        return H

    def _define_lindblad_operators(self):
        logging.info(
            f"Defining Lindblad operators for Runge-Kutta solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        L_ops = []

        for i in range(self.n_atoms):
            L = np.zeros((size, size), dtype=complex)
            for j in range(size // 2):
                L[j, j + size // 2] = np.sqrt(self.decay_rate)
            L_ops.append(L)
        return L_ops

    def _lindblad_rhs(self, t, rho_vec, H, L_ops):
        logging.debug(
            f"Evaluating RHS of Lindblad equation at time {t} with {self.n_atoms} atoms."
        )
        rho = rho_vec.reshape(H.shape)
        commutator = -1j * (H @ rho - rho @ H)
        dissipator = sum(
            L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            for L in L_ops
        )
        return (commutator + dissipator).ravel()

    def solve(self):
        logging.info(
            f"Solving Lindblad equation with Runge-Kutta solver with {self.n_atoms} atoms."
        )
        H = self._define_hamiltonian()
        L_ops = self._define_lindblad_operators()
        rho0 = np.zeros((2**self.n_atoms, 2**self.n_atoms))
        rho0[0, 0] = 1.0
        rho0_vec = rho0.ravel()

        dt = self.t_list[1] - self.t_list[0]
        result = [rho0_vec]

        for t in self.t_list[:-1]:
            k1 = dt * self._lindblad_rhs(t, rho0_vec, H, L_ops)
            k2 = dt * self._lindblad_rhs(t + dt / 2, rho0_vec + k1 / 2, H, L_ops)
            k3 = dt * self._lindblad_rhs(t + dt / 2, rho0_vec + k2 / 2, H, L_ops)
            k4 = dt * self._lindblad_rhs(t + dt, rho0_vec + k3, H, L_ops)

            rho0_vec = rho0_vec + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            result.append(rho0_vec)

        return np.array(result)


# Backward Euler Solver
class BackwardEulerLindbladSolver(LindbladSolver):
    def __init__(self, t_list, n_atoms, coupling, decay_rate):
        super().__init__(t_list, n_atoms)
        self.coupling = coupling
        self.decay_rate = decay_rate

    def _define_hamiltonian(self):
        logging.info(
            f"Defining Hamiltonian for Backward Euler solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        H = np.zeros((size, size), dtype=complex)

        for j in range(self.n_atoms):
            for k in range(j + 1, self.n_atoms):
                op_j = [qeye(2) if i != j else sigmap() for i in range(self.n_atoms)]
                op_k = [qeye(2) if i != k else sigmam() for i in range(self.n_atoms)]
                H += self.coupling * tensor(op_j).full() @ tensor(op_k).full()
        return H

    def _define_lindblad_operators(self):
        logging.info(
            f"Defining Lindblad operators for Backward Euler solver with {self.n_atoms} atoms."
        )
        size = 2**self.n_atoms
        L_ops = []

        for i in range(self.n_atoms):
            L = np.zeros((size, size), dtype=complex)
            for j in range(size // 2):
                L[j, j + size // 2] = np.sqrt(self.decay_rate)
            L_ops.append(L)
        return L_ops

    def solve(self):
        logging.info(
            f"Solving Lindblad equation with Backward Euler solver with {self.n_atoms} atoms."
        )
        H = self._define_hamiltonian()
        L_ops = self._define_lindblad_operators()
        size = 2**self.n_atoms

        rho0 = np.zeros((size, size), dtype=complex)
        rho0[0, 0] = 1.0  # Initial state
        rho = rho0

        dt = self.t_list[1] - self.t_list[0]
        identity = np.eye(size, dtype=complex)
        result = [rho]

        for t in self.t_list[:-1]:
            lhs = identity - dt * (-1j * H + sum(L.conj().T @ L for L in L_ops))
            rhs = rho @ identity  # Ensure `rhs` has the correct shape
            rho = np.linalg.solve(lhs, rhs).reshape(size, size)  # Solve and reshape
            result.append(rho)

        return np.array(result)
