API Reference
====================

This page contains the complete API reference for LLMize.

Core Classes
------------

OptimizationResult
~~~~~~~~~~~~~~~~~~~

.. autoclass:: llmize.base.OptimizationResult
   :members:
   :undoc-members:

Optimizer
~~~~~~~~~

.. autoclass:: llmize.base.Optimizer
   :members:
   :undoc-members:

Optimizers
----------

OPRO
~~~~

.. autoclass:: llmize.methods.opro.OPRO
   :members:
   :undoc-members:
   :show-inheritance:

ADOPRO
~~~~~~

.. autoclass:: llmize.methods.adopro.ADOPRO
   :members:
   :undoc-members:
   :show-inheritance:

HLMEA
~~~~~

.. autoclass:: llmize.methods.hlmea.HLMEA
   :members:
   :undoc-members:
   :show-inheritance:

HLMSA
~~~~~

.. autoclass:: llmize.methods.hlmsa.HLMSA
   :members:
   :undoc-members:
   :show-inheritance:

Callbacks
---------

EarlyStopping
~~~~~~~~~~~~~

.. autoclass:: llmize.callbacks.early_stopping.EarlyStopping
   :members:
   :undoc-members:

AdaptTempOnPlateau
~~~~~~~~~~~~~~~~~~

.. autoclass:: llmize.callbacks.adapt_temp_on_plateau.AdaptTempOnPlateau
   :members:
   :undoc-members:

OptimalScoreStopping
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: llmize.callbacks.optimal_score_stopping.OptimalScoreStopping
   :members:
   :undoc-members:

Configuration
-------------

Config
~~~~~~

.. autoclass:: llmize.config.Config
   :members:
   :undoc-members:

Utility Modules
---------------

LLM Interface
~~~~~~~~~~~~~

.. automodule:: llmize.llm.llm_call
   :members:
   :undoc-members:

.. automodule:: llmize.llm.llm_init
   :members:
   :undoc-members:

Parsing Utilities
~~~~~~~~~~~~~~~~~

.. automodule:: llmize.utils.parsing
   :members:
   :undoc-members:

.. automodule:: llmize.utils.truncate
   :members:
   :undoc-members:

Logging
~~~~~~~

.. automodule:: llmize.utils.logger
   :members:
   :undoc-members:

Submodules
----------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules/llmize.methods
   modules/llmize.utils
   modules/llmize.callbacks 