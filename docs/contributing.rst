Contributing
====================

We welcome contributions to LLMize! This guide will help you get started.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

    git clone https://github.com/yourusername/llmize.git
    cd llmize

3. Create a virtual environment:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

.. code-block:: bash

    pip install -e ".[dev]"

5. Install pre-commit hooks:

.. code-block:: bash

    pre-commit install

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run with coverage
    pytest --cov=llmize
    
    # Run specific test categories
    pytest -m "not long_test"  # Skip API tests
    pytest -m "long_test"      # Run only API tests

Code Style
~~~~~~~~~~

LLMize follows PEP 8 and uses these tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Format your code with:

.. code-block:: bash

    black llmize/
    isort llmize/
    flake8 llmize/
    mypy llmize/

How to Contribute
-----------------

Reporting Bugs
~~~~~~~~~~~~~~

1. Check existing issues first
2. Create a new issue with:
   - Clear title describing the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, LLMize version)
   - Code snippet if applicable

Suggesting Features
~~~~~~~~~~~~~~~~~~~

1. Open an issue with "Feature:" prefix
2. Describe the use case
3. Explain why it would be valuable
4. Consider implementation approach

Submitting Pull Requests
~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork and create a feature branch:

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. Make your changes:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation
   - Include type hints

3. Test your changes:

.. code-block:: bash

    pytest
    mypy llmize/

4. Commit and push:

.. code-block:: bash

    git commit -m "feat: add new feature description"
    git push origin feature/your-feature-name

5. Open a Pull Request with:
   - Clear title and description
   - Reference any related issues
   - Screenshots if UI changes
   - Testing instructions

Development Guidelines
----------------------

Code Organization
~~~~~~~~~~~~~~~~~

- Follow the existing module structure
- Use descriptive names for functions and variables
- Add docstrings to all public functions and classes
- Include type hints for better IDE support

Documentation
~~~~~~~~~~~~~

- Update docstrings for any API changes
- Add examples for new features
- Update the README if needed
- Consider adding to the FAQ

Testing
~~~~~~~

- Write unit tests for new functionality
- Use descriptive test names
- Mock external API calls when possible
- Test both success and error cases

- Use markers for different test types:
  - ``@pytest.mark.long_test`` for tests requiring API calls
  - ``@pytest.mark.very_long_test`` for extensive tests

Specific Contribution Areas
---------------------------

New Optimizers
~~~~~~~~~~~~~~

To add a new optimizer:

1. Create a new file in ``llmize/methods/``
2. Inherit from the ``Optimizer`` base class
3. Implement the ``optimize`` method
4. Add comprehensive docstrings
5. Include example usage
6. Add tests

Example structure:

.. code-block:: python

    from ..base import Optimizer, OptimizationResult
    
    class MyOptimizer(Optimizer):
        """Description of your optimizer."""
        
        def __init__(self, problem_text=None, obj_func=None, ...):
            """Initialize your optimizer."""
            super().__init__(problem_text, obj_func, ...)
        
        def optimize(self, ...):
            """Implement your optimization algorithm."""
            # Implementation here
            return OptimizationResult(...)

New Callbacks
~~~~~~~~~~~~~

To add a new callback:

1. Create a new file in ``llmize/callbacks/``
2. Implement ``on_step_end`` method
3. Add documentation and tests

Example:

.. code-block:: python

    class MyCallback:
        """Description of your callback."""
        
        def on_step_end(self, step, logs=None):
            """Called at the end of each optimization step."""
            # Implementation here
            return False  # Return True to stop optimization

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

We especially welcome:
- Tutorial improvements
- Additional examples
- Better explanations
- Translation to other languages

Community
---------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Use inclusive language
- Focus on constructive feedback
- Help others learn and grow

Getting Help
~~~~~~~~~~~~

If you need help contributing:

- Ask questions in GitHub Discussions
- Tag maintainers in issues
- Join our community channels
- Email: rizki@bwailabs.com

Recognition
~~~~~~~~~~~

All contributors are recognized in:
- AUTHORS file
- Release notes
- GitHub contributors list

Thank you for contributing to LLMize! 🎉 