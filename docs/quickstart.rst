Quick Start
====================

This guide will help you get started with LLMize in just a few minutes.

Installation
------------

First, install LLMize using pip:

.. code-block:: bash

    pip install llmize

Then, set up your API key as an environment variable:

.. code-block:: bash

    # For Google Gemini
    export GEMINI_API_KEY="your-api-key-here"
    
    # For OpenRouter
    export OPENROUTER_API_KEY="your-openrouter-api-key"
    
    # For Hugging Face
    export HUGGINGFACE_API_KEY="your-huggingface-api-key"

Your First Optimization
------------------------

Here's a simple example of how to use LLMize with OPRO approach to minimize a quadratic function:

.. code-block:: python

    from llmize import OPRO
    import os

    # Define your objective function
    def obj_func(x):
        if isinstance(x, list):
            return (float(x[0]) + 2)**2  # Minimum at x=-2
        else:
            return (float(x) + 2)**2  # Minimum at x=-2

    # Create an optimizer instance
    opro = OPRO(
        problem_text="Minimize (x+2)^2",
        obj_func=obj_func,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Provide initial samples and their scores
    init_samples = ["0", "1", "-1"]
    init_scores = [4, 9, 1]  # (0+2)^2, (1+2)^2, (-1+2)^2

    # Run the optimization
    result = opro.minimize(
        init_samples=init_samples,
        init_scores=init_scores,
        num_steps=2,
        batch_size=2
    )

    # Access results
    print(f"Best solution: {result.best_solution}")
    print(f"Best score: {result.best_score}")
    print(f"Convergence history: {result.best_score_history}")
    print(f"Per-step scores: {result.best_score_per_step}")

Multi-dimensional Optimization
-------------------------------

LLMize also supports multi-dimensional optimization:

.. code-block:: python

    from llmize import OPRO

    def sphere_function(x):
        """Minimize sum(x_i^2) - minimum at origin"""
        return sum(float(i)**2 for i in x)

    opro = OPRO(
        problem_text="Minimize the sphere function sum(x_i^2) for 3 dimensions",
        obj_func=sphere_function,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Initial samples as lists for multi-dimensional problems
    init_samples = [["1", "1", "1"], ["2", "0", "0"], ["0", "2", "0"]]
    init_scores = [3, 4, 4]  # sum of squares

    result = opro.minimize(
        init_samples=init_samples,
        init_scores=init_scores,
        num_steps=5,
        batch_size=3
    )

    print(f"Best solution: {result.best_solution}")
    print(f"Best score: {result.best_score}")

Maximization Problems
---------------------

For maximization, simply use the `maximize()` method:

.. code-block:: python

    def neg_sphere(x):
        """Maximize -sum(x_i^2) - maximum at origin"""
        return -sum(float(i)**2 for i in x)

    opro = OPRO(
        problem_text="Maximize -sum(x_i^2)",
        obj_func=neg_sphere,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    result = opro.maximize(
        init_samples=[["1", "1"], ["2", "2"]],
        init_scores=[-2, -8],
        num_steps=5
    )

Using Different Optimizers
--------------------------

Choose the optimizer based on your problem:

.. code-block:: python

    from llmize import OPRO, ADOPRO, HLMEA, HLMSA

    # Simple problems - OPRO
    opro = OPRO(problem_text="...", obj_func=func, api_key=key)

    # Complex landscapes - ADOPRO
    adopro = ADOPRO(problem_text="...", obj_func=func, api_key=key)

    # Combinatorial problems - HLMEA
    hlmea = HLMEA(problem_text="...", obj_func=func, api_key=key)

    # Multi-modal problems - HLMSA
    hlmsa = HLMSA(problem_text="...", obj_func=func, api_key=key)

Next Steps
----------

- Check out the :doc:`examples` for more detailed tutorials
- Learn about :doc:`configuration` to customize defaults
- Explore the :doc:`api` for full documentation
- Read about :doc:`advanced_usage` for callbacks and parallel processing 