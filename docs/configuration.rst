Configuration
============

The optimizer can be configured with various parameters:

.. code-block:: python

    opro = OPRO(
        problem_text="Your problem description",
        obj_func=your_objective_function,
        api_key=your_api_key,
        temperature=0.7,
        max_tokens=100,
        # ... other parameters
    )

Dependencies
-----------

- Python >= 3.8
- numpy >= 1.21.0
- google-generativeai >= 0.3.0
- colorama >= 0.4.6
- matplotlib >= 3.5.0 