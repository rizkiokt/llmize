Frequently Asked Questions
==========================

This section addresses common issues and questions when using LLMize.

General Questions
-----------------

**Q: Which optimizer should I use?**

A: The choice depends on your problem:
- **OPRO**: Best for simple problems with clear patterns
- **ADOPRO**: Good for complex landscapes where OPRO gets stuck
- **HLMEA**: Ideal for combinatorial optimization (TSP, scheduling)
- **HLMSA**: Suitable for multi-modal problems with many local optima

See the optimizer comparison table in the README for more details.

**Q: How many initial samples do I need?**

A: Generally, 3-10 initial samples are sufficient. The number should:
- Be at least 3 to give the LLM enough context
- Scale with problem complexity (more for complex problems)
- Consider your API budget (each sample costs an evaluation)

**Q: What temperature should I use?**

A: Temperature controls randomness in LLM outputs:
- **0.0-0.3**: For precise, deterministic problems
- **0.7-1.0**: Default, good balance of exploration/exploitation
- **1.0-2.0**: For highly creative exploration

Use the AdaptTempOnPlateau callback to adjust dynamically.

Common Issues
-------------

**Issue: "API key not found" error**

Solution:
1. Set the environment variable: ``export GEMINI_API_KEY="your-key"``
2. Or pass it directly: ``OPRO(api_key="your-key", ...)``
3. Or add to your ``.env`` file: ``GEMINI_API_KEY=your-key``

**Issue: Optimization converges too quickly to a local optimum**

Solutions:
1. Try ADOPRO instead of OPRO for adaptive prompting
2. Increase temperature to encourage exploration
3. Use HLMEA for better diversity maintenance
4. Check if your initial samples are diverse enough

**Issue: "Rate limit exceeded" errors**

Solutions:
1. Increase ``retry_delay`` in configuration
2. Reduce ``batch_size`` to make fewer API calls
3. Use ``parallel_n_jobs=1`` to avoid concurrent requests
4. Upgrade your API plan for higher limits

**Issue: Solutions are not valid (wrong format, constraints violated)**

Solutions:
1. Be more specific in ``problem_text`` about constraints
2. Include penalty functions in your objective function
3. Provide better initial examples showing valid formats
4. Use callbacks to monitor and reject invalid solutions

**Issue: Optimization is too slow**

Solutions:
1. Increase ``parallel_n_jobs`` for parallel evaluation
2. Reduce ``num_steps`` and ``batch_size``
3. Use a faster LLM model
4. Cache expensive objective function evaluations

Performance Tips
----------------

**Reducing API Costs**

1. Start with smaller ``num_steps`` (10-20) for testing
2. Use ``parallel_n_jobs`` to evaluate multiple solutions per API call
3. Implement early stopping to avoid unnecessary steps
4. Cache results for repeated evaluations

**Improving Convergence**

1. Provide diverse, high-quality initial samples
2. Write clear, specific problem descriptions
3. Use appropriate callbacks (EarlyStopping, AdaptTempOnPlateau)
4. Monitor ``best_score_history`` to diagnose issues

**Handling Large Problems**

1. Break large problems into smaller subproblems
2. Use HLMEA for combinatorial problems
3. Implement custom problem-specific operators
4. Consider dimensionality reduction techniques

Debugging Tips
--------------

**Enable Verbose Logging**

.. code-block:: python

    # Set verbose=2 for detailed output
    result = optimizer.optimize(verbose=2)

**Monitor Optimization Progress**

.. code-block:: python

    # Custom callback to track progress
    class ProgressTracker:
        def on_step_end(self, step, logs=None):
            print(f"Step {step}: Best score = {logs['best_score']}")
    
    result = optimizer.optimize(callbacks=[ProgressTracker()])

**Save and Load Results**

.. code-block:: python

    # Save results
    import json
    with open('results.json', 'w') as f:
        json.dump(result.to_dict(), f)
    
    # Load and analyze later
    with open('results.json', 'r') as f:
        data = json.load(f)
        # Reconstruct if needed

Getting Help
------------

If you're still having issues:

1. Check the :doc:`examples` for similar problems
2. Review the :doc:`api` reference
3. Search existing GitHub issues
4. Create a new issue with:
   - Your problem description
   - Code snippet
   - Error messages
   - Expected vs actual behavior

Community
---------

- GitHub Issues: https://github.com/rizkiokt/llmize/issues
- Email: rizki@bwailabs.com
