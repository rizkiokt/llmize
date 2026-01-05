# LLMize

LLMize is a Python package that uses Large Language Models (LLMs) for multipurpose, numerical optimization tasks. It provides a flexible and efficient framework for solving various optimization problems using LLM-based approaches, with support for both maximization and minimization objectives.

## Features

- **LLM-Based Optimization**: Utilizes LLMs for iteratively generating and optimizing solutions, inspired by OPRO methods [paper here](https://arxiv.org/abs/2309.03409)
- **Multiple LLM Providers**: Support for Google Gemini, OpenRouter (new in v0.3.0), and Hugging Face models
- **Multiple Optimizers**: Includes four powerful optimizers:
  - **OPRO**: Optimization by PROmpting - directly prompts LLMs to generate better solutions
  - **ADOPRO**: Adaptive OPRO - dynamically adjusts prompts based on optimization progress
  - **HLMEA**: Hyper-heuristic LLM-driven Evolutionary Algorithm - uses evolutionary strategies with LLM guidance
  - **HLMSA**: Hyper-heuristic LLM-driven Simulated Annealing - combines simulated annealing with LLM optimization
- **Flexible Problem Definition**: Supports both text-based problem descriptions and objective functions
- **Configuration System**: Centralized configuration management with TOML files and environment variables
- **Parallel Processing**: Built-in support for parallel evaluation of solutions to speed up optimization
- **Callback System**: Extensible callback mechanism for monitoring and controlling the optimization process
- **Early Stopping**: Built-in early stopping mechanism to prevent overfitting and save API costs
- **Adaptive Temperature**: Dynamic LLM temperature adjustment based on optimization progress
- **Comprehensive Results**: Detailed optimization results including best scores, solution history, and convergence metrics

## Installation

To install LLMize, you can use pip:

```bash
pip install llmize
```

For development installation:

```bash
git clone https://github.com/rizkiokt/llmize.git
cd llmize
pip install -e .
```

### API Keys Setup

LLMize supports multiple LLM providers. Set up the API keys for the providers you want to use:

```bash
# For Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"

# For OpenRouter (access to multiple models)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# For Hugging Face
export HUGGINGFACE_API_KEY="your-huggingface-api-key"
```

## Available Optimizers

### OPRO (Optimization by PROmpting)

The original approach that directly prompts LLMs to generate better solutions based on previous examples. Best for:
- Simple optimization problems
- Problems with clear solution patterns
- Quick prototyping and testing

```python
from llmize import OPRO
import os

opro = OPRO(
    problem_text="Minimize (x+2)^2",
    obj_func=obj_func,
    api_key=os.getenv("GEMINI_API_KEY")  # or OPENROUTER_API_KEY
)
```

### ADOPRO (Adaptive OPRO)

An enhanced version of OPRO that dynamically adjusts prompts based on optimization progress. Best for:
- Problems requiring adaptive strategies
- Complex optimization landscapes
- When OPRO gets stuck in local optima

```python
from llmize import ADOPRO

adopro = ADOPRO(
    problem_text="Optimize complex function",
    obj_func=complex_func,
    api_key=os.getenv("GEMINI_API_KEY")  # or OPENROUTER_API_KEY
)
```

### HLMEA (Hyper-heuristic LLM-driven Evolutionary Algorithm)

Uses evolutionary strategies with LLM guidance to maintain diversity and avoid premature convergence. Best for:
- Combinatorial optimization problems
- Large search spaces
- Problems requiring diverse solutions

```python
from llmize import HLMEA

hlmea = HLMEA(
    problem_text="Solve traveling salesman problem",
    obj_func=tsp_objective,
    api_key=os.getenv("GEMINI_API_KEY")  # or OPENROUTER_API_KEY
)
```

### HLMSA (Hyper-heuristic LLM-driven Simulated Annealing)

Combines simulated annealing principles with LLM optimization for controlled exploration. Best for:
- Problems with many local optima
- Fine-tuning solutions
- Temperature-sensitive optimization

```python
from llmize import HLMSA

hlmsa = HLMSA(
    problem_text="Find global minimum",
    obj_func=multimodal_func,
    api_key=os.getenv("GEMINI_API_KEY")  # or OPENROUTER_API_KEY
)
```

## Choosing the Right Optimizer

| Optimizer | Best For | Complexity | Diversity | Convergence Speed |
|-----------|----------|------------|-----------|------------------|
| OPRO | Simple problems | Low | Low | Fast |
| ADOPRO | Adaptive problems | Medium | Medium | Medium |
| HLMEA | Combinatorial/Large spaces | High | High | Slow |
| HLMSA | Multi-modal problems | High | Medium | Medium |

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

## LLM Provider Support

LLMize supports multiple LLM providers for your optimization tasks:

### Google Gemini

```python
# Using Google Gemini models
opro = OPRO(
    problem_text="Minimize (x+2)^2",
    obj_func=obj_func,
    llm_model="gemini-2.0-flash-thinking-exp",  # or gemini-1.5-pro, gemma-3-27b-it
    api_key=os.getenv("GEMINI_API_KEY")
)
```

### OpenRouter (New in v0.3.0)

```python
# Using OpenRouter to access various models
opro = OPRO(
    problem_text="Minimize (x+2)^2",
    obj_func=obj_func,
    llm_model="openrouter/anthropic/claude-3.5-sonnet",  # or any OpenRouter model
    api_key=os.getenv("OPENROUTER_API_KEY")
)
```

OpenRouter provides access to multiple models including:
- OpenAI: `openrouter/openai/gpt-4o`, `openrouter/openai/gpt-4o-mini`
- Anthropic: `openrouter/anthropic/claude-3.5-sonnet`, `openrouter/anthropic/claude-3.5-haiku`
- Google: `openrouter/google/gemini-2.0-flash-exp`
- Meta: `openrouter/meta-llama/llama-3.1-405b-instruct`
- And many more...

### Hugging Face

```python
# Using Hugging Face models
opro = OPRO(
    problem_text="Minimize (x+2)^2",
    obj_func=obj_func,
    llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key=os.getenv("HUGGINGFACE_API_KEY")
)
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

#### Available Callbacks

1. **EarlyStopping**: Stop optimization when no improvement is seen
```python
from llmize.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='best_score',  # Metric to monitor
    min_delta=0.01,        # Minimum change to qualify as improvement
    patience=10,           # Steps with no improvement before stopping
    verbose=1              # Print messages when early stopping triggers
)
```

2. **AdaptTempOnPlateau**: Reduce temperature when optimization plateaus
```python
from llmize.callbacks import AdaptTempOnPlateau

adapt_temp = AdaptTempOnPlateau(
    monitor='best_score',  # Metric to monitor
    factor=0.5,            # Factor to multiply temperature by
    patience=5,            # Steps with no improvement before reducing temp
    min_temp=0.1,          # Minimum temperature value
    verbose=1              # Print messages when temperature changes
)
```

3. **OptimalScoreStopping**: Stop when reaching a target score
```python
from llmize.callbacks import OptimalScoreStopping

optimal_stop = OptimalScoreStopping(
    optimal_score=0.99,     # Target score to reach
    tolerance=0.01,         # Tolerance for reaching target
    verbose=1              # Print messages when target is reached
)
```

#### Using Multiple Callbacks

```python
from llmize.callbacks import EarlyStopping, AdaptTempOnPlateau, OptimalScoreStopping

callbacks = [
    EarlyStopping(patience=10),
    AdaptTempOnPlateau(factor=0.5),
    OptimalScoreStopping(optimal_score=0.99, tolerance=0.01)
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

The `OptimizationResult` class provides comprehensive optimization results:

```python
# Access optimization results
print(f"Best solution: {results.best_solution}")
print(f"Best score: {results.best_score}")
print(f"Score history: {results.best_score_history}")
print(f"Per-step best scores: {results.best_score_per_step}")
print(f"Per-step average scores: {results.avg_score_per_step}")

# Convert to dictionary for serialization
results_dict = results.to_dict()
```

### Plotting Results

Visualize optimization progress:

```python
import matplotlib.pyplot as plt

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(results.best_score_history, label='Best Score')
plt.plot(results.avg_score_per_step, label='Average Score')
plt.xlabel('Step')
plt.ylabel('Score')
plt.title('Optimization Progress')
plt.legend()
plt.grid(True)
plt.show()
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

- Python >= 3.11
- numpy >= 1.21.0
- google-genai >= 1.15.0
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
