Examples
====================

This section provides various examples demonstrating how to use LLMize for different optimization problems.

Convex Optimization
----------------------

This example demonstrates how to use LLMize for convex optimization with constraints. It shows how to formulate convex problems and use different optimizers to find optimal solutions.

`convex_opt.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/convex_optimization/convex_opt.ipynb>`_

**Key concepts covered:**
- Formulating convex optimization problems
- Using OPRO for simple convex problems
- Constraint handling in LLM-based optimization

Traveling Salesman Problem
-----------------------------

This example shows how to solve the Traveling Salesman Problem using LLMize. The full example can be found in the notebook:

`tsp.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/tsp/tsp.ipynb>`_

**Key concepts covered:**
- Combinatorial optimization with LLMs
- Using HLMEA for TSP
- Maintaining solution diversity
- Tour representation and evaluation

Neural Network Hyperparameter Tuning
---------------------------------------

This example demonstrates hyperparameter optimization for neural networks, showing how LLMize can optimize complex, high-dimensional search spaces.

`mnist_tf.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/nn_hp_tuning/mnist_tf.ipynb>`_

**Key concepts covered:**
- Hyperparameter space definition
- Using ADOPRO for adaptive hyperparameter tuning
- Parallel evaluation of neural network configurations
- Early stopping to prevent overfitting

Linear Programming
---------------------

This example shows how to solve linear programming problems using LLMize, demonstrating the framework's versatility across different problem domains.

`lp_optimization.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/linear_programming/lp_optimization.ipynb>`_

**Key concepts covered:**
- Linear problem formulation
- Constraint representation
- Solution interpretation for linear problems

Nuclear Fuel Optimization
----------------------------

This example demonstrates a real-world application of LLMize for optimizing nuclear fuel parameters in a boiling water reactor (BWR). This is a complex, high-stakes optimization problem.

`nuclear_fuel_optimization.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/nuclear_fuel_optimization/bwr_ge14_opt.ipynb>`_

**Key concepts covered:**
- Complex engineering optimization
- Domain-specific constraints and requirements
- Using HLMSA for fine-tuning critical parameters
- Safety considerations in optimization

Choosing the Right Example
---------------------------

Based on your use case:

- **Getting started**: Begin with the convex optimization example
- **Combinatorial problems**: Check the TSP example
- **Machine learning**: See the neural network hyperparameter tuning example
- **Engineering applications**: Review the nuclear fuel optimization example
- **Linear problems**: Start with the linear programming example

Running the Examples
---------------------

To run any example:

1. Install LLMize: `pip install llmize`
2. Set up your API key as an environment variable
3. Clone the repository and navigate to the example directory
4. Open and run the Jupyter notebook

For more detailed examples and results, please refer to the examples directory in the repository.
