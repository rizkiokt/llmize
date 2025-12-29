# LLMize

LLMize is a Python package that uses Large Language Models (LLMs) for multipurpose, numerical optimization tasks. It provides a flexible and efficient framework for solving various optimization problems using LLM-based approaches, with support for both maximization and minimization objectives.

## Features

- **LLM-Based Optimization**: Utilizes LLM for iteratively generating and optimizing solutions, inspired by OPRO methods [paper here](https://arxiv.org/abs/2309.03409)
- **Multiple Optimizers**: Includes OPRO, ADOPRO, HLMEA, and HLMSA optimizers for different problem types
- **Flexible Problem Definition**: Supports both text-based problem descriptions and objective functions
- **Configuration System**: Centralized configuration management with TOML files and environment variables
- **Parallel Processing**: Built-in support for parallel evaluation of solutions
- **Callback System**: Extensible callback mechanism for monitoring and controlling the optimization process
- **Early Stopping**: Built-in early stopping mechanism to prevent overfitting
- **Adaptive Temperature**: Dynamic LLM temperature adjustment based on optimization progress
- **Comprehensive Results**: Detailed optimization results including best scores, solution history, and convergence metrics

## Installation

To install LLMize, you can use pip:

```bash
pip install llmize
```

For development installation:

```bash
git clone https://github.com/yourusername/llmize.git
cd llmize
pip install -e .
```

## Quick Start

Here's a simple example of how to use LLMize with OPRO approach:

```python
from llmize import OPRO
import os

def obj_func(x):
    if isinstance(x, list):
        return (float(x[0]) + 2)**2  # Minimum at x=-2
    else:
        return (float(x) + 2)**2  # Minimum at x=-2

opro = OPRO(
    problem_text="Minimize (x+2)^2",
    obj_func=obj_func,
    api_key=os.getenv("GEMINI_API_KEY")
)

init_samples = ["0", "1", "-1"]
init_scores = [4, 9, 1]  # (0+2)^2, (1+2)^2, (-1+2)^2

result = opro.minimize(
    init_samples=init_samples,
    init_scores=init_scores,
    num_steps=2,
    batch_size=2
)

# Access results using the new OptimizationResult class
print(f"Best solution: {result.best_solution}")
print(f"Best score: {result.best_score}")
print(f"Convergence history: {result.best_score_history}")
print(f"Per-step scores: {result.best_score_per_step}")
```

## Examples

The package includes several example implementations:

- **Neural Network Hyperparameter Tuning**: Optimize neural network architectures and hyperparameters
- **Traveling Salesman Problem**: Solve TSP using LLM-based optimization
- **Linear Programming**: Solve linear programming problems
- **Convex Optimization**: Handle convex optimization tasks
- **Nuclear Fuel Optimization**: Complex optimization in nuclear engineering [paper here](https://arxiv.org/abs/2503.19620)

Check the `examples/` directory for detailed implementations.

## Advanced Usage

### Callbacks

LLMize supports custom callbacks for monitoring and controlling the optimization process:

```python
from llmize.callbacks import EarlyStopping, AdaptTempOnPlateau

callbacks = [
    EarlyStopping(patience=5),
    AdaptTempOnPlateau(factor=0.5)
]

results = optimizer.maximize(
    callbacks=callbacks,
    # ... other parameters
)
```

### Parallel Processing

Enable parallel evaluation of solutions:

```python
results = optimizer.maximize(
    parallel_n_jobs=4,  # Number of parallel processes
    # ... other parameters
)
```

### Result Analysis

The new `OptimizationResult` class provides comprehensive optimization results:

```python
# Access optimization results
print(f"Best solution: {results.best_solution}")
print(f"Best score: {results.best_score}")
print(f"Score history: {results.best_score_history}")
print(f"Per-step scores: {results.best_score_per_step}")
print(f"Average scores: {results.avg_score_per_step}")
print(f"Number of steps: {results.num_steps}")
print(f"Total time: {results.total_time} seconds")
```

## Configuration

LLMize uses a flexible configuration system that allows you to customize defaults without modifying code. Configuration can be provided through:

### 1. Configuration File (TOML)

Create a `llmize.toml` file in your project root:

```toml
[llm]
# Default LLM model to use
default_model = "gemma-3-27b-it"

# Default temperature for LLM generation (0.0 to 2.0)
temperature = 1.0

# Maximum number of retries when hitting rate limits
max_retries = 10

# Delay between retries in seconds
retry_delay = 5

[optimization]
# Default number of optimization steps
default_num_steps = 50

# Default batch size for generating solutions
default_batch_size = 5

# Default number of parallel jobs for evaluation
parallel_n_jobs = 1
```

### 2. Environment Variables

Override configuration using environment variables:

```bash
export LLMIZE_DEFAULT_MODEL="gemini-2.0-flash-thinking-exp"
export LLMIZE_TEMPERATURE=0.8
export LLMIZE_MAX_RETRIES=15
```

### 3. In Code

You can still override defaults when creating optimizers:

```python
from llmize import OPRO

# Uses all defaults from config
opro = OPRO(
    problem_text="Your problem",
    obj_func=your_function,
    api_key=your_key
)

# Override specific parameters
opro = OPRO(
    problem_text="Your problem",
    obj_func=your_function,
    llm_model="custom-model",  # Override config
    api_key=your_key
)
```

### Configuration Priority

Settings are applied in the following priority (highest first):
1. Direct parameters in method calls
2. Environment variables
3. Configuration file
4. Default values

## Advanced Configuration

The optimizer can be configured with various parameters:

```python
opro = OPRO(
    problem_text="Your problem description",
    obj_func=your_objective_function,
    api_key=your_api_key,
    temperature=0.7,
    max_tokens=100,
    # ... other parameters
)
```

## Dependencies

- Python >= 3.8
- numpy >= 1.21.0
- google-genai>=1.15.0
- colorama >= 0.4.6
- matplotlib >= 3.5.0
- python-dotenv >= 0.19.0
- toml >= 0.10.2

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LLMize in your research, please cite:

```bibtex
@software{llmize2025,
  author = {M. R. Oktavian},
  company = {Blue Wave AI Labs},
  title = {LLMize: LLM-based Optimization Library for Python},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rizkiokt/llmize}
}
```

## Contact

For questions, suggestions, or support, please contact:
- Email: rizki@bwailabs.com
- GitHub Issues: [Open an issue](https://github.com/rizkiokt/llmize/issues)
