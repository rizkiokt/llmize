import numpy as np

from ..base import Optimizer, OptimizationResult
from ..llm.llm_init import initialize_llm
from ..utils.parsing import parse_pairs
from ..utils.truncate import truncate_pairs
from ..utils.logger import log_info, log_warning, log_error, log_critical, log_debug
from ..callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau

class OPRO(Optimizer):
    """
    :no-index:
    OPRO optimizer for optimizing tasks using a specified LLM model.

    This class inherits from the `Optimizer` class and allows configuration 
    of various parameters related to the optimization process.

    :param str llm_model: The name of the LLM model to use (default from config).
    :param str api_key: The API key for accessing the model (default: None).
    :param int num_steps: The number of optimization steps (default: 50).
    :param int batch_size: The batch size used for optimization (default: 5).
    """

    def __init__(self, problem_text=None, obj_func=None, llm_model=None, api_key=None):
        """
        Initialize the OPRO optimizer with the provided configuration.
        Inherits from `Optimizer`.

        :param str llm_model: The name of the LLM model to use.
        :param str api_key: The API key for accessing the model.
        :param int num_steps: The number of optimization steps.
        :param int batch_size: The batch size used for optimization.
        """
        super().__init__(problem_text=problem_text, obj_func=obj_func, llm_model=llm_model, api_key=api_key)
    

    def meta_prompt(self, batch_size, example_pairs, optimization_type="maximize"):
        """
        Generate a prompt for the LLM model to generate new solutions.
        Parameters:
        - batch_size (int): Number of new solutions to generate.
        - example_pairs (str): Example solutions and scores.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").

        Returns:
        - text: A formatted prompt structure.
        """
        example_texts = """Below are some examples of solutions and their scores:"""

        if optimization_type == "maximize":
            text1 = "higher"
            text2 = "highest"
        else:
            text1 = "lower"
            text2 = "lowest"

        instruction = f"""
Generate exactly {batch_size} new solutions that:
- Are distinct from all previous solutions.
- Have {text1} scores than the {text2} provided.
- Respect the relationships based on logical reasoning.

Each solution should start with <sol> and end with </sol> with a comma between parameters. 
Make sure the length of solutions match examples given. Don't guess for the scores as they will be calculated by an objective function.
"""
        prompt = "\n".join([self.problem_text, example_texts, example_pairs, instruction])

        return prompt
    
    def optimize(self, init_samples=None, init_scores=None, num_steps=None, batch_size=None,
                 temperature=None, callbacks=None, verbose=1, optimization_type="maximize", parallel_n_jobs=None):
        
        """
        Run the OPRO optimization algorithm.

        Parameters:
        - init_samples (list): A list of initial solutions.
        - init_scores (list): A list of initial scores corresponding to init_samples.
        - num_steps (int): The number of optimization steps (default from config).
        - batch_size (int): The number of new solutions to generate at each step (default from config).
        - temperature (float): The temperature for the LLM model (default from config).
        - callbacks (list): A list of callback functions to be triggered at the end of each step.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").
        - parallel_n_jobs (int): Number of parallel jobs for evaluation (default from config).

        Returns:
        - results (OptimizationResult): An object containing the optimization results.
        """
        from ..config import get_config
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

        client = initialize_llm(self.llm_model, self.api_key)

        if verbose > 0: 
            log_info(f"Running OPRO optimization with {num_steps} steps and batch size {batch_size}...")
        best_solution = None
        if optimization_type == "maximize":
            best_score = np.max(init_scores)
        elif optimization_type == "minimize":
            best_score = np.min(init_scores)
        else:
            log_critical("Invalid optimization_type. Choose 'maximize' or 'minimize'.")
            raise ValueError("optimization_type must be 'maximize' or 'minimize'")
        
        best_score_history = [best_score]
        avg_score_per_step = [np.average(init_scores)]
        best_score_per_step = [best_score]

        #max_examples = batch_size

        # Call the helper function to initialize callbacks
        self._initialize_callbacks(callbacks, temperature)

        for step in range(num_steps+1):
            if step == 0:
                if verbose > 0: 
                    log_info(f"Step {step} - Best Initial Score: {best_score:.3f}, Average Initial Score: {np.average(init_scores):.3f}")
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
                continue
            
            if verbose > 1: 
                log_debug(f"Example pairs: {example_pairs}")

            prompt = self.meta_prompt(batch_size, example_pairs, optimization_type)


            solution_array = self._generate_solutions(client, prompt, temperature, 
                                                            batch_size, verbose)
            
            best_score, best_solution, step_scores, best_step_score = self._evaluate_solutions(solution_array, best_solution,
                                                                              optimization_type, verbose, best_score, parallel_n_jobs)
            new_pairs = parse_pairs(solution_array, step_scores)
            example_pairs = example_pairs + new_pairs
            #example_pairs = truncate_pairs(example_pairs, max_examples, optimization_type)

            avg_step_score = sum(step_scores) / len(solution_array)
            best_score_per_step.append(best_step_score)
            avg_score_per_step.append(avg_step_score)
            best_score_history.append(best_score)
            if verbose > 0: 
                log_info(f"Step {step} - Current Best Score: {best_score:.3f}, Average Batch Score: {avg_step_score:.3f} - Best Batch Score: {best_step_score:.3f}")

            # Callbacks: Trigger at the end of each step
            if callbacks:
                for callback in callbacks:
                    logs = {callback.monitor: best_score}  # Pass logs with the monitored metric
                    new_temperature = callback.on_step_end(step, logs)  # Callback could adjust the temperature
                    if new_temperature is not None:
                        temperature = new_temperature  # Update temperature if needed
                    # Check if early stopping is triggered
                    early_stop = isinstance(callback, EarlyStopping) and callback.wait >= callback.patience
                    optimal_stop = isinstance(callback, OptimalScoreStopping) and callback.on_step_end(step, logs)
                    if early_stop or optimal_stop:                        
                        break
                if early_stop or optimal_stop:
                    break

        return OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            best_score_history=best_score_history,
            best_score_per_step=best_score_per_step,
            avg_score_per_step=avg_score_per_step
        )


