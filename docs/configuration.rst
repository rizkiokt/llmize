Configuration
=============

LLMize uses a flexible configuration system that allows you to customize defaults without modifying code.

Configuration Sources
---------------------

Configuration is loaded from multiple sources in order of priority:

1. **Direct parameters** (highest priority)
2. **Environment variables**
3. **Configuration files**
4. **Default values** (lowest priority)

Configuration Files
-------------------

LLMize looks for configuration files in the following locations:

1. ``~/.llmize/config.toml`` (user's home directory)
2. ``/etc/llmize/config.toml`` (system-wide)
3. ``llmize.toml`` (project root)
4. ``.llmize.toml`` (project root, hidden)

Example Configuration
---------------------

Here's an example ``llmize.toml`` configuration file:

.. code-block:: toml

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

Configuration Options
---------------------

LLM Settings
~~~~~~~~~~~~

- ``default_model`` (str): Default LLM model to use (default: "gemma-3-27b-it")
- ``temperature`` (float): Default temperature for LLM generation (default: 1.0)
- ``max_retries`` (int): Maximum retry attempts for failed API calls (default: 10)
- ``retry_delay`` (int): Delay between retries in seconds (default: 5)

Optimization Settings
~~~~~~~~~~~~~~~~~~~~

- ``default_num_steps`` (int): Default number of optimization steps (default: 50)
- ``default_batch_size`` (int): Default batch size for solution generation (default: 5)
- ``parallel_n_jobs`` (int): Default number of parallel jobs for evaluation (default: 1)

Environment Variables
---------------------

All configuration options can be overridden using environment variables with the ``LLMIZE_`` prefix:

.. code-block:: bash

    export LLMIZE_DEFAULT_MODEL="gemini-2.0-flash-thinking-exp"
    export LLMIZE_TEMPERATURE=0.8
    export LLMIZE_MAX_RETRIES=15
    export LLMIZE_RETRY_DELAY=3
    export LLMIZE_DEFAULT_NUM_STEPS=100
    export LLMIZE_DEFAULT_BATCH_SIZE=10
    export LLMIZE_PARALLEL_N_JOBS=4

Using Configuration in Code
----------------------------

Accessing Configuration
~~~~~~~~~~~~~~~~~~~~~~~

You can access the current configuration programmatically:

.. code-block:: python

    from llmize.config import get_config
    
    config = get_config()
    print(f"Default model: {config.default_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max retries: {config.max_retries}")

Reloading Configuration
~~~~~~~~~~~~~~~~~~~~~~~

To reload configuration from a specific file:

.. code-block:: python

    from llmize.config import reload_config
    
    # Reload from custom file
    reload_config("/path/to/custom/config.toml")

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from llmize import OPRO
    
    # Uses all defaults from config
    optimizer = OPRO(
        problem_text="Minimize x^2",
        obj_func=lambda x: float(x)**2,
        api_key="your-api-key"
    )

Overriding Defaults
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Override only the model, use other defaults
    optimizer = OPRO(
        problem_text="Minimize x^2",
        obj_func=lambda x: float(x)**2,
        llm_model="custom-model",  # Override config
        api_key="your-api-key"
    )
    
    # Override in optimize method
    result = optimizer.optimize(
        init_samples=["1", "2"],
        init_scores=[1, 4],
        num_steps=100,  # Override config default
        batch_size=10   # Override config default
    )

Project-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``llmize.toml`` in your project root:

.. code-block:: toml

    [llm]
    default_model = "gemini-2.0-flash-thinking-exp"
    temperature = 0.7
    
    [optimization]
    default_num_steps = 200
    default_batch_size = 10

Now all optimizers in your project will use these defaults.

User-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create ``~/.llmize/config.toml`` for user-wide defaults:

.. code-block:: bash

    mkdir -p ~/.llmize
    cat > ~/.llmize/config.toml << EOF
    [llm]
    default_model = "your-preferred-model"
    temperature = 0.5
    EOF
