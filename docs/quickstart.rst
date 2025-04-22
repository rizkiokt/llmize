Quick Start
====================

Here's a simple example of how to use LLMize with OPRO approach:

.. code-block:: python

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