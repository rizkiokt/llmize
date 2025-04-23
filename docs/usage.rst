Usage
====================

Basic Usage
------------

Here's how to use LLMize for convex optimization with different initialization methods:

1. First, import the necessary modules:

.. code-block:: python

    import numpy as np
    from llmize import OPRO
    import os

2. Define the objective function:

.. code-block:: python

    def objective_convex_penalty(x):
        """
        Objective function for the Convex Optimization problem with penalties.
        The function is minimized.
        """
        x1, x2 = x
        f = (x1 - 3)**2 + (x2 + 2)**2 + np.sin(x1 + x2) + 4
        
        # Constraint violations
        penalty = 0
        large_penalty = 1e6  # Large penalty value

        if x1 < 0 or x1 > 5:
            penalty += large_penalty
        if x2 < 0 or x2 > 5:
            penalty += large_penalty

        return f + penalty

3. Define the problem description:

.. code-block:: python

    problem_text = """Problem: Convex Optimization
    -----------------------------------------------------
    Objective: Minimize the function
        f(x1, x2) = (x1 - 3)^2 + (x2 + 2)^2 + sin(x1 + x2) + 4

    Subject to constraints:
        0 ≤ x1 ≤ 5
        0 ≤ x2 ≤ 5
    """

4. Initialize the optimizer:

.. code-block:: python

    optimizer = OPRO(
        problem_text=problem_text,
        obj_func=objective_convex_penalty,
        llm_model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY")
    )

5. Generate initial samples:

.. code-block:: python

    # Generate initial samples
    num_samples = 4
    batch_size = num_samples**2
    x1_range = np.linspace(0, 4.5, num_samples)
    x2_range = np.linspace(0, 4.5, num_samples)
    solutions = []
    scores = []
    
    for i in range(num_samples):
        for j in range(num_samples):
            solutions.append([x1_range[i], x2_range[j]])
            scores.append(objective_convex_penalty([x1_range[i], x2_range[j]]))

6. Run the optimization:

.. code-block:: python

    results = optimizer.minimize(
        init_samples=solutions,
        init_scores=scores,
        num_steps=250,
        batch_size=16
    )

7. Plot the results:

.. code-block:: python

    from llmize.utils.plotting import plot_scores
    plot_scores(results)

Advanced Usage
---------------

This section demonstrates advanced features for controlling the optimization process. For the full example, see:
`convex_opt.ipynb <https://github.com/rizkiokt/llmize/blob/main/examples/convex_optimization/convex_opt.ipynb>`_

1. Using Callbacks:

.. code-block:: python

    from llmize.callbacks import (
        EarlyStopping,
        AdaptTempOnPlateau,
        OptimalScoreStopping
    )

    # Define callbacks
    earlystop_callback = EarlyStopping(
        monitor='best_score',
        min_delta=0.001,
        patience=50,
        verbose=1
    )

    optimal_score_callback = OptimalScoreStopping(
        optimal_score=7.90,
        tolerance=0.01
    )

    adapt_temp_callback = AdaptTempOnPlateau(
        monitor='best_score',
        init_temperature=1.0,
        min_delta=0.001,
        patience=20,
        factor=1.1,
        max_temperature=1.9,
        verbose=1
    )

    # Combine callbacks
    callbacks = [earlystop_callback, optimal_score_callback, adapt_temp_callback]

    # Run optimization with callbacks
    results = optimizer.minimize(
        init_samples=solutions,
        init_scores=scores,
        num_steps=250,
        batch_size=16,
        callbacks=callbacks
    )

For more detailed examples and results, please refer to the :doc:`examples` section.

See Also
---------------
* :doc:`advanced_usage` for more advanced features
* :doc:`configuration` for configuration options
* :doc:`api` for complete API reference 