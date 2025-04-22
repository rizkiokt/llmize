Advanced Usage
=============

Callbacks
--------

LLMize supports custom callbacks for monitoring and controlling the optimization process:

.. code-block:: python

    from llmize.callbacks import EarlyStopping, AdaptTempOnPlateau

    callbacks = [
        EarlyStopping(patience=5),
        AdaptTempOnPlateau(factor=0.5)
    ]

    results = optimizer.maximize(
        callbacks=callbacks,
        # ... other parameters
    )

Parallel Processing
-----------------

Enable parallel evaluation of solutions:

.. code-block:: python

    results = optimizer.maximize(
        parallel_n_jobs=4,  # Number of parallel processes
        # ... other parameters
    )

Result Analysis
-------------

The new ``OptimizationResult`` class provides comprehensive optimization results:

.. code-block:: python

    # Access optimization results
    print(f"Best solution: {results.best_solution}")
    print(f"Best score: {results.best_score}")
    print(f"Score history: {results.best_score_history}")
    print(f"Per-step scores: {results.best_score_per_step}")
    print(f"Average scores: {results.avg_score_per_step}")
    print(f"Number of steps: {results.num_steps}")
    print(f"Total time: {results.total_time} seconds") 