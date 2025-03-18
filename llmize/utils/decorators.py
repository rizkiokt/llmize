import time
from llmize.utils.logger import log_info, log_debug, log_critical

def check_init(func):
    """
    A decorator to validate the initialization parameters for optimization functions.

    This decorator checks the validity of the parameters `init_samples`, `init_scores`,
    `num_steps`, and `batch_size` before calling the decorated function. If any of these
    parameters are invalid, an appropriate `ValueError` is raised.

    Parameters:
    - func (callable): The function to be wrapped and validated.

    Raises:
    - ValueError: If `init_samples` or `init_scores` is `None`.
    - ValueError: If `num_steps` or `batch_size` is not a positive integer.
    """

    def inner(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None, verbose=1, parallel_n_jobs=1):

        if init_samples is None or init_scores is None:
            log_critical("init_samples and init_scores must be provided.")
            raise ValueError("init_samples and init_scores must be provided.")
        if num_steps <= 0 or not isinstance(num_steps, int):
            log_critical("num_steps must be a positive integer.")
            raise ValueError("num_steps must be a positive integer.")
        if batch_size <= 0 or not isinstance(batch_size, int):
            log_critical("batch_size must be a positive integer.")  
            raise ValueError("batch_size must be a positive integer.")
        
        # Check for temperature
        if not isinstance(temperature, float):
            log_critical("temperature must be a float.")
            raise ValueError("temperature must be a float.")
        # Check for verbose
        if not isinstance(verbose, int):
            log_critical("verbose must be an integer.")
            raise ValueError("verbose must be an integer.")

        return func(self, init_samples, init_scores, num_steps, batch_size, temperature, callbacks, verbose, parallel_n_jobs)
    return inner

def time_it(func):
    """
    A decorator that measures and logs the execution time of a function.

    This decorator logs the time taken by the decorated function to execute
    using the `log_info` method. It records the start and end time of the
    function execution and calculates the duration in seconds.

    Parameters:
    - func (callable): The function whose execution time is to be measured.

    Returns:
    - callable: The wrapped function with time logging capability.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record end time
        log_info(f"Execution time of {func.__name__}: {end_time - start_time:.3f} seconds")
        return result
    return wrapper
