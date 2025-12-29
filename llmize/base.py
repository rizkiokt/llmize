import functools
import multiprocessing as mp
from .config import get_config
from .utils.parsing import parse_pairs
from .llm.llm_call import generate_content
from .llm.llm_init import initialize_llm
from .utils.logger import log_info, log_critical, log_debug, log_warning, log_error
from .utils.parsing import parse_response
from .utils.decorators import check_init, time_it
from .callbacks import EarlyStopping, AdaptTempOnPlateau

class OptimizationResult:
    """
    A class to store the results of an optimization process.
    
    Attributes:
        best_solution: The best solution found during optimization
        best_score: The score of the best solution
        best_score_history: List of best scores at each step
        best_score_per_step: List of best scores in each batch
        avg_score_per_step: List of average scores in each batch
    """
    def __init__(self, best_solution, best_score, best_score_history, best_score_per_step, avg_score_per_step):
        self.best_solution = best_solution
        self.best_score = best_score
        self.best_score_history = best_score_history
        self.best_score_per_step = best_score_per_step
        self.avg_score_per_step = avg_score_per_step

    def to_dict(self):
        """Convert the result to a dictionary format."""
        return {
            "best_solution": self.best_solution,
            "best_score": self.best_score,
            "best_score_history": self.best_score_history,
            "best_score_per_step": self.best_score_per_step,
            "avg_score_per_step": self.avg_score_per_step
        }

class Optimizer:
    def __init__(self, problem_text=None, obj_func=None, llm_model=None, api_key=None):
        """
        Initialize the Optimizer with the general configuration.    
        """
        config = get_config()
        
        self.problem_text = problem_text
        self.obj_func = obj_func
        self.llm_model = llm_model if llm_model is not None else config.default_model
        self.api_key = api_key

    @check_init
    @time_it
    def maximize(self, init_samples=None, init_scores=None, num_steps=None, batch_size=None,
                 temperature=None, callbacks=None, verbose=1, parallel_n_jobs=None):
        """
        Run the optimization algorithm to maximize the objective function.

        Parameters:
        - init_samples (list): A list of initial solutions.
        - init_scores (list): A list of initial scores corresponding to init_samples.
        - num_steps (int): The number of optimization steps (default from config).
        - batch_size (int): The batch size used for optimization (default from config).
        - temperature (float): The temperature for the LLM model (default from config).
        - callbacks (list): A list of callback functions to be triggered at the end of each step.
        - verbose (int): The verbosity level (default: 1).
        - parallel_n_jobs (int): The number of parallel jobs for evaluation (default from config).

        Returns:
        - results (OptimizationResult): An object containing the optimization results.
        """
        config = get_config()
        
        # Use config defaults if not provided
        if num_steps is None:
            num_steps = config.default_num_steps
        if batch_size is None:
            batch_size = config.default_batch_size
        if temperature is None:
            temperature = config.temperature
        if parallel_n_jobs is None:
            parallel_n_jobs = config.parallel_n_jobs
        
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature, 
                              callbacks=callbacks, verbose=verbose, optimization_type="maximize", parallel_n_jobs=parallel_n_jobs)
    
    @check_init
    @time_it
    def minimize(self, init_samples=None, init_scores=None, num_steps=None, batch_size=None,
                 temperature=None, callbacks=None, verbose=1, parallel_n_jobs=None):
        """
        Run the optimization algorithm to minimize the objective function.

        Parameters:
        - init_samples (list): A list of initial solutions.
        - init_scores (list): A list of initial scores corresponding to init_samples.
        - num_steps (int): The number of optimization steps (default from config).
        - batch_size (int): The batch size used for optimization (default from config).
        - temperature (float): The temperature for the LLM model (default from config).
        - callbacks (list): A list of callback functions to be triggered at the end of each step.
        - verbose (int): The verbosity level (default: 1).
        - parallel_n_jobs (int): The number of parallel jobs for evaluation (default from config).

        Returns:
        - results (dict): A dictionary containing the best solution, best score, best score history, best score per step, and average score per step.
        """
        config = get_config()
        
        # Use config defaults if not provided
        if num_steps is None:
            num_steps = config.default_num_steps
        if batch_size is None:
            batch_size = config.default_batch_size
        if temperature is None:
            temperature = config.temperature
        if parallel_n_jobs is None:
            parallel_n_jobs = config.parallel_n_jobs
            
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature,
                              callbacks=callbacks, verbose=verbose, optimization_type="minimize", parallel_n_jobs=parallel_n_jobs)
    
    def get_configuration(self):
        """
        Returns the general configuration of the optimizer.
        """
        return {
            "llm_model": self.llm_model,
            "problem_text": self.problem_text,
            "obj_func": (self.obj_func.func.__name__ if isinstance(self.obj_func, functools.partial) 
                        else self.obj_func.__name__) if self.obj_func else None
        }

    def meta_prompt(self, batch_size, example_pairs, optimization_type="maximize"):
        """
        Dummy function for meta_prompt. Should be overridden by subclasses.
        
        Parameters:
        - batch_size (int): Number of new solutions to generate.
        - example_pairs (str): Example solutions and scores.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").

        Returns:
        - prompt (str): A formatted prompt structure.
        """
        return "prompt"
    
    
    def get_sample_prompt(self, batch_size=None, optimization_type="maximize", init_samples=None, init_scores=None):
        """
        Generate a sample prompt for the language model based on the provided parameters.

        Parameters:
        - batch_size (int): The number of new solutions to generate (default from config).
        - optimization_type (str): The type of optimization to perform, either "maximize" or "minimize" (default: "maximize").
        - init_samples (list): A list of initial solutions (default: None).
        - init_scores (list): A list of initial scores corresponding to init_samples (default: None).

        Returns:
        - str: The generated prompt as a string to be used for generating solutions from the language model.
        """
        if batch_size is None:
            batch_size = get_config().default_batch_size
            
        example_pairs = parse_pairs(init_samples, init_scores)

        prompt = self.meta_prompt(batch_size=batch_size, example_pairs=example_pairs, optimization_type=optimization_type)

        return prompt

    def get_sample_response(self, prompt):

        """
        Generate a response from the language model using the provided prompt.

        Parameters:
        - prompt (str): The prompt to be used for generating the response.

        Returns:
        - str: The generated content as a string from the language model.
        """

        client = initialize_llm(self.llm_model, self.api_key)

        return generate_content(client, self.llm_model, prompt)

    def _generate_solutions(self, client, prompt, temperature, batch_size, verbose, max_retries=5, hp_parse=False):
        """
        Generate solutions by retrying content generation until the solution array has the expected batch size.
        
        Parameters:
            client: The client instance to be passed to the content generation function.
            llm_model: The language model to be used for generating content.
            prompt: The prompt to send to the model.
            temperature: The temperature parameter for content generation.
            batch_size: Expected number of solutions.
            verbose: Verbosity level for debug logging.
            max_retries: Maximum number of retry attempts.
            hp_parse: Whether to parse the hyperparameters from the response.
        Returns:
            A list of solutions that matches the expected batch_size.
        
        Raises:
            ValueError: If the expected number of solutions cannot be generated after max_retries.
        """

        response = generate_content(client, self.llm_model, prompt, temperature)
        if hp_parse:
            solution_array, hp = parse_response(response, hp_parse)
            hp_none = hp is None
        else:
            solution_array = parse_response(response, hp_parse)
            hp_none = False

        if verbose > 2:
            #log_debug(f"Prompt: {prompt}")
            log_debug(f"Response: {response}")
        if verbose > 1: log_debug(f"Generated Solutions: {solution_array}")
        
        # Always show the response in debug mode to troubleshoot HLMSA issues
        if verbose >= 1 and solution_array is None:
            log_error(f"Failed to parse solutions. LLM response was: {response}")
        

        retry = 0
        
        while solution_array is None or len(solution_array) < batch_size or (hp_parse and hp_none):
            log_warning("Number of solutions parsed is less than batch size. Retrying...")
            response = generate_content(client, self.llm_model, prompt, temperature)
            if hp_parse:
                solution_array, hp = parse_response(response, hp_parse)
                hp_none = hp is None
            else:
                solution_array = parse_response(response, hp_parse)
                hp_none = False

            if verbose > 2: 
                log_debug(f"Response for retry {retry+1}: {response}")
            if verbose > 1:
                log_debug(f"Generated Solutions for retry {retry+1}: {solution_array}")

            retry += 1
            if retry >= max_retries:
                log_critical("Failed to generate solutions after multiple attempts.")
                raise ValueError("Failed to generate solutions after multiple attempts.")
        
        if len(solution_array) > batch_size:
            log_warning(f"Number of solutions parsed is greater than batch size. Removing extra solutions.")
            # Remove first extra solutions
            solution_array = solution_array[:batch_size]

        if hp_parse:
            return solution_array, hp
        else:
            return solution_array
    
    def _evaluate_solutions(self, solution_array, best_solution, optimization_type, verbose, best_score=None, parallel_n_jobs=None):
        """
        Evaluate a list of solutions and update the best solution based on an objective function.
        
        Parameters:
            solution_array (list): A list of solutions to evaluate.
            optimization_type (str): "maximize" or "minimize".
            verbose (int): Verbosity level for logging.
            obj_func (callable): A function that takes a solution and returns its score.
            best_score (float, optional): The current best score. If not provided, it will be
                initialized based on the optimization_type.
            parallel_n_jobs (int): Number of CPU cores to use for parallel evaluation.
                If 1 (default), uses sequential processing.
                If >1, uses parallel processing with specified number of cores.
                If -1, uses all available cores.
                
        Returns:
            tuple: (best_score, best_solution, step_scores, best_step_score)
                best_score (float): The updated best score after evaluating solutions.
                best_solution (any): The solution corresponding to the best score.
                step_scores (list): A list of scores for each solution in solution_array.
                best_step_score (float): The best score in the current batch.
        
        Raises:
            ValueError: If optimization_type is not "maximize" or "minimize".
        """
        self.optimization_type = optimization_type  # Store for _evaluate_single_solution
        
        # Use config default if parallel_n_jobs is None
        if parallel_n_jobs is None:
            parallel_n_jobs = get_config().parallel_n_jobs
        
        if optimization_type == "maximize":
            best_step_score = -float('inf')  # Start with the lowest possible value for maximization
            if best_score is None:
                best_score = -float('inf')
        elif optimization_type == "minimize":
            best_step_score = float('inf')   # Start with the highest possible value for minimization
            if best_score is None:
                best_score = float('inf')
        else:
            raise ValueError("optimization_type must be 'maximize' or 'minimize'")
        
        # Evaluate solutions either in parallel or sequentially
        if parallel_n_jobs != 1:
            try:
                # If parallel_n_jobs is -1, use all available cores
                if parallel_n_jobs == -1:
                    parallel_n_jobs = mp.cpu_count()
                # Ensure parallel_n_jobs is at least 1
                parallel_n_jobs = max(1, min(parallel_n_jobs, mp.cpu_count()))
                
                if verbose > 1:
                    log_debug(f"Using {parallel_n_jobs} CPU cores for parallel evaluation")
                    
                with mp.Pool(processes=parallel_n_jobs) as pool:
                    step_scores = pool.map(self._evaluate_single_solution, solution_array)
            except Exception as e:
                log_error(f"Parallel evaluation failed, falling back to sequential: {e}")
                step_scores = [self._evaluate_single_solution(solution) for solution in solution_array]
        else:
            step_scores = [self._evaluate_single_solution(solution) for solution in solution_array]

        # Process results
        for i, score in enumerate(step_scores):
            if verbose > 1:
                log_debug(f"Score for solution {solution_array[i]}: {score}")

            # Update best step score for the current batch
            if (optimization_type == "maximize" and score > best_step_score) or \
            (optimization_type == "minimize" and score < best_step_score):
                best_step_score = score

            # Update the overall best score and solution if found
            if (optimization_type == "maximize" and score > best_score) or \
            (optimization_type == "minimize" and score < best_score):
                best_score = score
                best_solution = solution_array[i]

        return best_score, best_solution, step_scores, best_step_score
    
    def _evaluate_single_solution(self, solution):
        """
        Evaluate a single solution using the objective function.
        
        Parameters:
            solution: The solution to evaluate
            
        Returns:
            float: The score of the solution
        """
        try:
            return self.obj_func(solution)
        except Exception as e:
            log_error(f"Error occurred while evaluating solution {solution}: {e}")
            return float('inf') if self.optimization_type == "minimize" else float('-inf')

    def _initialize_callbacks(self,callbacks, temperature):
        """
        Initialize wait counter and temperature for callbacks if early stopping or adaptive temperature is used.
        
        Parameters:
        - callbacks (list): A list of callback functions.
        - temperature (float): The initial temperature for the LLM model.
        """
        if callbacks:
            for callback in callbacks:
                if isinstance(callback, EarlyStopping):
                    callback.wait = 0
                if isinstance(callback, AdaptTempOnPlateau):
                    callback.temperature = temperature
                    callback.wait = 0