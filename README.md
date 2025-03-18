# LLMize

LLMize is a powerful Python package that leverages Large Language Models (LLMs) for optimization tasks. It provides a flexible and efficient framework for solving various optimization problems using LLM-based approaches, with support for both maximization and minimization objectives.

## Features

- **LLM-Based Optimization**: Utilizes state-of-the-art language models (default: Gemini 2.0 Flash) for generating and optimizing solutions
- **Flexible Problem Definition**: Supports both text-based problem descriptions and objective functions
- **Parallel Processing**: Built-in support for parallel evaluation of solutions
- **Callback System**: Extensible callback mechanism for monitoring and controlling the optimization process
- **Temperature Control**: Adjustable temperature parameter for controlling solution diversity
- **Early Stopping**: Built-in early stopping mechanism to prevent overfitting
- **Adaptive Temperature**: Dynamic temperature adjustment based on optimization progress

## Installation

To install LLMize, you can use pip:

```bash
pip install .
```

## Quick Start

Here's a simple example of how to use LLMize:

```python
from llmize import Optimizer

# Define your problem
problem_text = "Find the optimal values of x and y that maximize the function f(x,y) = x^2 + y^2"
objective_function = lambda x, y: x**2 + y**2

# Initialize the optimizer
optimizer = Optimizer(
    problem_text=problem_text,
    obj_func=objective_function,
    llm_model="gemini-2.0-flash",  # or your preferred model
    api_key="your-api-key"
)

# Run optimization
results = optimizer.maximize(
    num_steps=50,
    batch_size=5,
    temperature=1.0,
    verbose=1
)

# Access results
best_solution = results['best_solution']
best_score = results['best_score']
```

## Examples

The package includes several example implementations:

- **Neural Network Hyperparameter Tuning**: Optimize neural network architectures and hyperparameters
- **Traveling Salesman Problem**: Solve TSP using LLM-based optimization
- **Linear Programming**: Solve linear programming problems
- **Convex Optimization**: Handle convex optimization tasks
- **Nuclear Fuel Optimization**: Complex optimization in nuclear engineering

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

## Configuration

The optimizer can be configured with various parameters:

- `problem_text`: Text description of the optimization problem
- `obj_func`: Objective function to optimize
- `llm_model`: Choice of LLM model
- `api_key`: API key for the LLM service
- `num_steps`: Number of optimization iterations
- `batch_size`: Number of solutions to generate per step
- `temperature`: Controls solution diversity
- `verbose`: Level of logging output

## Contributing

We welcome contributions! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Report bugs or suggest features through issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, suggestions, or support, please contact:
- Email: rizki@bwailabs.com
- GitHub Issues: [Open an issue](https://github.com/yourusername/llmize/issues)
