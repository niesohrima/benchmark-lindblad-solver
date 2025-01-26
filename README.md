# Lindblad Equation Solver Benchmark

This project benchmarks various solvers for the Lindblad master equation, which models the dynamics of open quantum systems. The solvers compared include implementations using QuTiP, SciPy, Runge-Kutta, and Backward Euler methods. Parallelism has not been taken into account.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Solvers](#solvers)
- [Benchmarking](#benchmarking)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Lindblad master equation is a key mathematical model for describing the evolution of quantum systems interacting with their environment. This project provides an efficient benchmarking framework to compare different numerical solvers for the equation. While parallelism is required for modelling the evolution of large quantum systems, still choosing an optimal algorithm for scaling up for concurrent operation is necessary. This benchmark aims exactly at this problem. By benchmarking the algorithms, we can choose the more optimal approach for parallel implementation. Note the coupling and decay rate of the target system does impact the solvers' performance as well.

## Features
- Pairwise interaction between the atoms has been considered in the model.
- Spontanous emission has been included in the model.
- Multiple solvers for the Lindblad equation:
  - QuTiP-based solver
  - SciPy-based solver
  - Runge-Kutta solver
  - Backward Euler solver
- Scalability tests for systems with up to N atoms.
- Visualization of runtime and memory usage.
- Logging for detailed performance insights.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project_directory>
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script to perform scalability benchmarks:
   ```bash
   python main.py
   ```
2. Results will be logged to `benchmark-lindblad-solver.log` and visualized as plots.

## Solvers

The following solvers are implemented in `solvers.py`:

- **QuTiP Solver**: Uses QuTiP's `mesolve` function for solving the Lindblad equation.
- **SciPy Solver**: Implements a custom solver based on `solve_ivp` from SciPy.
- **Runge-Kutta Solver**: A manual implementation of the Runge-Kutta integration method.
- **Backward Euler Solver**: A backward Euler method tailored for the Lindblad equation.

## Benchmarking

The `Benchmark` class in `benchmark.py`:

- **Scalability Testing**: Evaluates solvers across varying the number of atoms in the system and measures runtime and memory usage.
- **Visualization**: Generates comparative plots for runtime and memory usage, saved as `benchmark-lindblad-solver.png`.

### Example Results

- **Runtime Plot**: Shows how solver runtimes scale with the number of atoms.
- **Memory Usage Plot**: Displays memory consumption trends across solvers.

## Tests

Unit tests are provided to validate the functionality of the solvers and benchmarking framework. To run the tests:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Use `pytest` to run all tests:
   ```bash
   pytest
   ```

3. Test files are located in the `tests/` directory and cover the following:
   - **Solvers**: Tests for individual solver implementations.
   - **Benchmarking**: Tests for the benchmarking framework.

Example command to run a specific test:
```bash
pytest tests/test_solvers.py
```

Test results will indicate whether all components of the project are functioning correctly.

## Contributing

We welcome contributions to improve the project. To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://mit-license.org/) file for details.
