import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import logging


class Benchmark:
    def __init__(self, max_atoms, time_span, samples_per_tau, coupling, decay_rate):
        """
        Initialize a Benchmark instance.

        Parameters:
            max_atoms (int): The maximum number of atoms in the system.
            time_span (float): The total time for the simulation as number of decay times.
            samples_per_tau (int): The number of samples per decay time.
            coupling (float): The coupling strength between the atoms.
            decay_rate (float): The spontaneous emission rate.
        """
        self.max_atoms = max_atoms
        self.time_span = time_span
        self.samples_per_tau = samples_per_tau
        self.coupling = coupling
        self.decay_rate = decay_rate
        self.scalability_results = {}

    def run(self, solver_name, solver_instance):
        """
        Measure the performance of a solver.

        The method measures the runtime and memory usage of the specified solver
        instance and returns the results in a dictionary.

        Parameters:
            solver_name (str): A string identifying the solver.
            solver_instance: An instance of the solver class.

        Returns:
            dict: A dictionary containing the solver name, runtime, and memory usage.
        """
        logging.info(f"Measuring performance for {solver_name}.")
        start_time = time.time()
        memory_profile = memory_usage((solver_instance.solve, (), {}))
        end_time = time.time()

        return {
            "runtime": end_time - start_time,
            "memory_usage": max(memory_profile),
        }

    def test_scalability(self, solver_class):
        """
        Measure the scalability of a solver.

        The method tests the solver's performance by executing it with an increasing
        number of atoms in the system. The results are stored in the `scalability_results`
        dictionary.

        Parameters:
            solver_class: The solver class to test.

        Returns:
            None
        """
        logging.info(f"Testing scalability for {solver_class.__name__}.")
        atom_counts = range(2, self.max_atoms + 1)
        scalability_metrics = {
            "runtime": [],
            "memory_usage": [],
            "atom_counts": list(atom_counts),
        }

        for num_atoms in atom_counts:
            logging.info(f"Testing {solver_class.__name__} with {num_atoms} atoms.")
            solver_instance = solver_class(
                self.time_span,
                self.samples_per_tau,
                num_atoms,
                self.coupling,
                self.decay_rate,
            )
            performance_metrics = self.run(
                f"{solver_class.__name__} ({num_atoms} atoms)", solver_instance
            )
            scalability_metrics["runtime"].append(performance_metrics["runtime"])
            scalability_metrics["memory_usage"].append(
                performance_metrics["memory_usage"]
            )

        self.scalability_results[solver_class.__name__] = scalability_metrics

    def display_results(self, export_to_file=True, output_file=f"benchmark-plots.png"):
        """
        Visualize and optionally export the scalability results.

        This method generates plots to visualize the runtime and memory usage of
        different solvers as a function of the number of atoms. It creates a single
        figure with two subplots: one for runtime and one for memory usage. Each
        solver's performance is represented as a line plot with markers. The plots
        include labels and titles for clarity, and a combined legend is displayed
        below the plots. If specified, the plots can be exported to a file.

        Parameters:
            export_to_file (bool): Whether to save the plots to a file. Defaults to True.
            output_file (str): The filename for the exported plot image. Defaults to "benchmark-plots.png".

        Returns:
            None
        """

        plot_height = 7
        plot_width = 7 * 1.618  # golden ratio

        plt.figure(figsize=(plot_width, plot_height))

        # Runtime plot
        runtime_axis = plt.subplot(1, 2, 1)
        for solver_name, metrics in self.scalability_results.items():
            runtime_axis.plot(
                metrics["atom_counts"],
                metrics["runtime"],
                marker="o",
                ls="--",
                alpha=0.7,
                label=solver_name,
            )
        runtime_axis.set_xlabel("Number of Atoms")
        runtime_axis.set_ylabel("Runtime (s)")
        runtime_axis.set_title("Runtime")
        runtime_axis.spines["top"].set_visible(False)
        runtime_axis.spines["right"].set_visible(False)

        # Memory usage plot
        memory_axis = plt.subplot(1, 2, 2)
        for solver_name, metrics in self.scalability_results.items():
            memory_axis.plot(
                metrics["atom_counts"],
                metrics["memory_usage"],
                marker="o",
                ls="--",
                alpha=0.7,
                label=solver_name,
            )
        memory_axis.set_xlabel("Number of Atoms")
        memory_axis.set_ylabel("Memory Usage (MiB)")
        memory_axis.set_title("Memory Usage")
        memory_axis.spines["top"].set_visible(False)
        memory_axis.spines["right"].set_visible(False)

        # Combined legend below the plots
        handles, labels = runtime_axis.get_legend_handles_labels()
        plt.figlegend(
            handles,
            labels,
            loc="lower center",
            ncol=len(self.scalability_results),
            frameon=False,
        )

        plt.suptitle(
            f"Benchmark solvers for decay rate {self.decay_rate:.2e} rad.s^-1 and coupling {self.coupling:.2e} rad.s^-1",
            fontsize=12,
            y=0.97,
        )

        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        if export_to_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
