import numpy as np
from qutip import *
from scipy.integrate import solve_ivp
import logging


# Base Solver Class
class LindbladSolver:
    def __init__(
        self, time_span, samples_per_decay_time, n_atoms, decay_rate, coupling
    ):
        """
        Initialize the Lindblad solver.

        Parameters:
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_decay_time (int): The number of samples per decay time.
            n_atoms (int): The number of atoms in the system.
            decay_rate (float): The spontaneous emission rate.
        """
        self.time_span = time_span
        self.samples_per_decay_time = samples_per_decay_time
        self.n_atoms = n_atoms
        self.decay_rate = decay_rate
        self.coupling = coupling

        # consider normarlization to the coupling rate for dissipative systems.
        self._normalize_coupling()

        self.t_list = self._generate_t_list()

    def _normalize_coupling(self):
        """
        Normalize the coupling strength and decay rate.

        This method normalizes the coupling strength to the decay rate by dividing the
        coupling strength by the decay rate and setting the decay rate to 1. This is
        useful for dissipative systems where the decay rate is non-zero.
        """

        if self.decay_rate != 0:
            self.coupling = self.coupling / self.decay_rate
            self.decay_rate = 1
        else:
            self.coupling = 1 # in the absence of decay, set coupling to 1

    def _generate_t_list(self):
        """
        Generates the time list based on the decay time and user-provided parameters.
        """
        total_time = self.time_span
        num_samples = int(self.samples_per_decay_time * self.time_span)
        return np.linspace(0, total_time, num_samples)

    def solve(self):
        raise NotImplementedError("Subclasses must implement the solve method.")


# QuTiP Solver
class QuTiPLindbladSolver(LindbladSolver):
    """
    Lindblad equation solver using the QuTiP library.

    This solver utilizes QuTiP's built-in tools to define the Hamiltonian, collapse operators,
    and solve the master equation. QuTiP is optimized for quantum systems.
    """

    def __init__(
        self, time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    ):
        """
        Initialize the QuTiP Lindblad solver.

        Parameters:
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_decay_time (int): The number of samples per decay time.
            n_atoms (int): The number of atoms in the system.
            coupling (float): The coupling strength between the atoms.
            decay_rate (float): The spontaneous emission rate.
        """
        super().__init__(
            time_span, samples_per_decay_time, n_atoms, decay_rate, coupling
        )

    def _define_hamiltonian(self):
        """
        Define the system's Hamiltonian.

        The Hamiltonian includes pairwise interactions between atoms, modeled as:
            H = \sum_{j < k} coupling * (\sigma_j^+ \sigma_k^- + \sigma_k^+ \sigma_j^-)
        where:
            - \sigma_j^+ and \sigma_j^- are the raising and lowering operators for the j-th atom.

        Returns:
            Qobj: Hamiltonian as a QuTiP quantum object.
        """
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
        """
        Define the collapse operators (dissipators).

        Each atom undergoes spontaneous emission, represented by the operator:
            C_j = sqrt(decay_rate) * \sigma_j^-
        where \sigma_j^- is the lowering operator for the j-th atom.

        Returns:
            list: List of collapse operators as QuTiP quantum objects.
        """
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
        """
        Define the initial state of the system.

        The initial state is a superposition of states where each atom can be excited or in the ground state.

        Returns:
            Qobj: Initial density matrix as a QuTiP quantum object.
        """
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
        """
        Solve the Lindblad master equation using QuTiP.

        Returns:
            Result: QuTiP result object containing the time evolution of the density matrix.
        """
        logging.info(
            f"Solving Lindblad equation with QuTiP solver with {self.n_atoms} atoms."
        )
        H = self._define_hamiltonian()
        c_ops = self._define_dissipators()
        rho0 = self._define_initial_state()
        return mesolve(H, rho0, self.t_list, c_ops)


# SciPy Solver
class SciPyLindbladSolver(LindbladSolver):
    """
    Lindblad equation solver using SciPy's ODE integration tools.

    This solver uses explicit integration of the Lindblad master equation.
    """

    def __init__(
        self, time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    ):
        """
        Initialize the SciPy Lindblad solver.

        Parameters:
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_decay_time (int): The number of samples per decay time.
            n_atoms (int): The number of atoms in the system.
            coupling (float): The coupling strength between the atoms.
            decay_rate (float): The spontaneous emission rate.
        """
        super().__init__(
            time_span, samples_per_decay_time, n_atoms, decay_rate, coupling
        )

    def _define_hamiltonian(self):
        """
        Define the system's Hamiltonian.

        The Hamiltonian includes pairwise interactions between atoms, modeled as:
            H = \sum_{j < k} coupling * (\sigma_j^+ \sigma_k^- + \sigma_k^+ \sigma_j^-)

        Returns:
            np.ndarray: Hamiltonian as a NumPy array.
        """
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
        """
        Define the collapse operators (Lindblad operators).

        Each atom undergoes spontaneous emission, represented by the operator:
            L_j = sqrt(decay_rate) * \sigma_j^-

        Returns:
            list: List of Lindblad operators as NumPy arrays.
        """
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
        """
        Compute the right-hand side of the Lindblad equation.

        Args:
            t (float): Current time.
            rho_vec (np.ndarray): Flattened density matrix.
            H (np.ndarray): Hamiltonian.
            L_ops (list): Lindblad operators.

        Returns:
            np.ndarray: Time derivative of the density matrix, flattened.
        """
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
        """
        Solve the Lindblad master equation using SciPy.

        Returns:
            OdeResult: SciPy result object containing the solution.
        """
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
    """
    Lindblad equation solver using a custom Runge-Kutta integration method.

    This solver explicitly implements the 4th-order Runge-Kutta method for integrating
    the Lindblad master equation.
    """

    def __init__(
        self, time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    ):
        """
        Initialize the Runge-Kutta Lindblad solver.

        Parameters:
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_decay_time (int): The number of samples per decay time.
            n_atoms (int): The number of atoms in the system.
            coupling (float): The coupling strength between the atoms.
            decay_rate (float): The spontaneous emission rate.
        """
        super().__init__(
            time_span, samples_per_decay_time, n_atoms, decay_rate, coupling
        )

    def _define_hamiltonian(self):
        """
        Define the system's Hamiltonian.

        Returns:
            np.ndarray: Hamiltonian as a NumPy array.
        """
        size = 2**self.n_atoms
        H = np.zeros((size, size), dtype=complex)
        for j in range(self.n_atoms):
            for k in range(j + 1, self.n_atoms):
                op_j = [qeye(2) if i != j else sigmap() for i in range(self.n_atoms)]
                op_k = [qeye(2) if i != k else sigmam() for i in range(self.n_atoms)]
                H += self.coupling * tensor(op_j).full() @ tensor(op_k).full()
        return H

    def _define_lindblad_operators(self):
        """
        Define the collapse operators (Lindblad operators).

        Returns:
            list: List of Lindblad operators as NumPy arrays.
        """
        size = 2**self.n_atoms
        L_ops = []
        for i in range(self.n_atoms):
            L = np.zeros((size, size), dtype=complex)
            for j in range(size // 2):
                L[j, j + size // 2] = np.sqrt(self.decay_rate)
            L_ops.append(L)
        return L_ops

    def _lindblad_rhs(self, t, rho_vec, H, L_ops):
        """
        Compute the right-hand side of the Lindblad equation.

        Returns:
            np.ndarray: Time derivative of the density matrix, flattened.
        """
        rho = rho_vec.reshape(H.shape)
        commutator = -1j * (H @ rho - rho @ H)
        dissipator = sum(
            L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            for L in L_ops
        )
        return (commutator + dissipator).ravel()

    def solve(self):
        """
        Solve the Lindblad master equation using a 4th-order Runge-Kutta method.

        Returns:
            np.ndarray: Time-evolved density matrices.
        """
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
    """
    Lindblad equation solver using the Backward Euler integration method.

    This solver uses an implicit scheme to integrate the Lindblad master equation,
    suitable for stiff systems.
    """

    def __init__(
        self, time_span, samples_per_decay_time, n_atoms, coupling, decay_rate
    ):
        """
        Initialize the Backward Euler Lindblad solver.

        Parameters:
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_decay_time (int): The number of samples per decay time.
            n_atoms (int): The number of atoms in the system.
            coupling (float): The coupling strength between the atoms.
            decay_rate (float): The spontaneous emission rate.
        """
        super().__init__(
            time_span, samples_per_decay_time, n_atoms, decay_rate, coupling
        )

    def _define_hamiltonian(self):
        """
        Define the system's Hamiltonian.

        Returns:
            np.ndarray: Hamiltonian as a NumPy array.
        """
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
        """
        Define the collapse operators (Lindblad operators).

        Returns:
            list: List of Lindblad operators as NumPy arrays.
        """
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
        """
        Solve the Lindblad master equation using Backward Euler method.

        Returns:
            np.ndarray: Time-evolved density matrices.
        """
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
