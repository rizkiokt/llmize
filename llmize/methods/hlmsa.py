import numpy as np

from ..base import Optimizer, OptimizationResult
from ..llm.llm_init import initialize_llm
from ..utils.parsing import parse_pairs
from ..utils.truncate import truncate_pairs
from ..utils.logger import log_info, log_warning, log_error, log_critical, log_debug
from ..callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau

class HLMSA(Optimizer):
    """
    :no-index:
    HLMSA: Hyper-heuristic LLM-driven Simulated Annealing
    HLMSA optimizer for optimizing tasks using a specified LLM model.

    This class inherits from the `Optimizer` class and allows configuration 
    of various parameters related to the optimization process.

    :param str llm_model: The name of the LLM model to use (default from config).
    :param str api_key: The API key for accessing the model (default: None).
    :param int num_steps: The number of optimization steps (default: 50).
    :param int batch_size: The batch size used for optimization (default: 5).
    """

    def __init__(self, problem_text=None, obj_func=None, llm_model=None, api_key=None):
        """
        Initialize the HLMSA optimizer with the provided configuration.
        Inherits from `Optimizer`.

        :param str llm_model: The name of the LLM model to use.
        :param str api_key: The API key for accessing the model.
        :param int num_steps: The number of optimization steps.
        :param int batch_size: The batch size used for optimization.
        """
        super().__init__(problem_text=problem_text, obj_func=obj_func, llm_model=llm_model, api_key=api_key)
    

    def meta_prompt(self, batch_size, example_pairs, optimization_type="maximize", 
                    hp_text="The solutions below are generated randomly."):
        """
        Generate a prompt for the LLM model to generate new solutions using Simulated Annealing.
        Parameters:
        - batch_size (int): Number of new solutions to generate.
        - example_pairs (str): Example solutions and scores.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").

        Returns:
        - text: A formatted prompt structure.
        """

        backstory = """
You are a hyper-heuristic LLM-driven optimization algorithm using Simulated Annealing to explore and refine solutions.  
You dynamically adapt the cooling rate and perturbation strategies to balance exploration and exploitation.  

## Problem Description  
You need to optimize a given problem by evolving a set of candidate solutions. The objective is to iteratively improve solutions while maintaining diversity and avoiding local optima.  
"""

        example_texts = """Below are some examples of solutions and their scores from the previous population:"""

        if optimization_type == "maximize":
            text1 = "higher"
            text2 = "highest"
        else:
            text1 = "lower"
            text2 = "lowest"

        instruction = f"""
The best solution has the {text2} score from the provided population.

## Your Task  

Generate **exactly {batch_size} diverse solutions** for the next iteration using Simulated Annealing.  

### **Step 1: Adaptive Cooling Rate Selection**  
- Choose an appropriate **cooling rate (α)** to control the acceptance probability of worse solutions.  
- A **higher cooling rate (e.g., α → 1)** maintains exploration, while a **lower cooling rate (e.g., α → 0.8)** increases exploitation.  
- Balance between exploration and exploitation dynamically based on the quality and diversity of previous solutions.  

### **Step 2: Solution Perturbation (Neighborhood Search)**  
- Generate new candidate solutions by applying one or more perturbation strategies:  
  - **Small Perturbation** (e.g., minor tweaks for fine-tuning).  
  - **Large Perturbation** (e.g., drastic changes to escape local optima).  
  - **Problem-Specific Perturbations**.
- Choose perturbation strength based on the cooling rate and current solution landscape.  

### **Step 3: Uniqueness Enforcement**  
- Ensure that each generated solution is **unique** compared to previous and current ones.  
- If a duplicate is found, **apply additional perturbations** until uniqueness is achieved.  

## **Output Format**  

Return exactly **{batch_size} unique solutions** and the chosen hyperparameters in the following format:  

### **Hyperparameters Output**  

<hp> cooling_rate <\\\\hp>

### **Solutions Output**  

<sol> param1, param2, ..., paramn <\\\\sol>

**Only provide the solutions and hyperparameters—do not include any extra text. Do not include any code.**  
"""
        prompt = "\n".join([backstory, self.problem_text, example_texts, hp_text, example_pairs, instruction])

        return prompt

    def _accept_solutions(self, prev_solutions, prev_scores, solution_array, step_scores, sa_temperature, optimization_type="maximize"):

        current_solutions = []
        current_scores = []

        for i in range(len(solution_array)):
            delta_e = step_scores[i] - prev_scores[i]
            if optimization_type == "maximize":
                accept_better = delta_e > 0
            else:
                accept_better = delta_e < 0
            acceptance_probability = np.exp(-delta_e/sa_temperature)
            if np.random.rand() < acceptance_probability or accept_better:
                current_solutions.append(solution_array[i])
                current_scores.append(step_scores[i])
            else:
                current_solutions.append(prev_solutions[i])
                current_scores.append(prev_scores[i])

        return current_solutions, current_scores    
    
    def optimize(self, init_samples=None, init_scores=None, num_steps=None, batch_size=None,
                 temperature=None, callbacks=None, verbose=1, optimization_type="maximize", parallel_n_jobs=None):
        
        """
        Run the HLMSA optimization algorithm.

        Parameters:
        - init_samples (list): A list of initial solutions.
        - init_scores (list): A list of initial scores corresponding to init_samples.
        - num_steps (int): The number of optimization steps (default: 50).
        - batch_size (int): The number of new solutions to generate at each step (default: 5).
        - temperature (float): The temperature for the LLM model (default: 1.0).
        - callbacks (list): A list of callback functions to be triggered at the end of each step.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").

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
            log_info(f"Running HLMSA optimization with {num_steps} steps and batch size {batch_size}...")
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

        init_sa_temperature = 1000 #initial temperature
        final_sa_temperature = 1.0 #final temperature
        cooling_rate = 0.95 #initial cooling rate

        # Call the helper function to initialize callbacks
        self._initialize_callbacks(callbacks, temperature)

        for step in range(num_steps+1):
            if step == 0:
                sa_temperature = init_sa_temperature
                if verbose > 0: 
                    log_info(f"Step {step} - SA Temperature: {sa_temperature:.2f} - Best Initial Score: {best_score:.3f}, Average Initial Score: {np.average(init_scores):.3f}")
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
                hp_text = "The solutions below are generated randomly."
                prev_solutions, prev_scores = init_samples, init_scores
                continue
            
            if verbose > 1: 
                log_debug(f"Example pairs: {example_pairs}")    

            prompt = self.meta_prompt(batch_size, example_pairs, optimization_type, hp_text)
            if verbose > 3:
                log_debug(f"Prompt: {prompt}")

            solution_array, hp = self._generate_solutions(client, prompt, temperature, 
                                                            batch_size, verbose, hp_parse=True)
            
            # If hyperparameter parsing failed but we got solutions, try again without hp_parse
            if solution_array is not None and hp is None and len(solution_array) >= batch_size:
                log_warning("Hyperparameter parsing failed, proceeding with default cooling rate")
                hp = [0.95]  # Default cooling rate
            

            # Check if hp is not None and has the correct format
            if hp is None or hp[0] < 0 or hp[0] > 1:
                log_warning("Invalid or missing hyperparameters format.")
                hp_text = "The cooling rate used in previous step are unknown."
            else:
                hp_text = f"""The hyperparameter (cooling rate) used in previous step is: <hp> {hp} <\\\\hp>"""
                cooling_rate = hp[0]

            best_score, best_solution, step_scores, best_step_score = self._evaluate_solutions(solution_array, best_solution,
                                                                              optimization_type, verbose, best_score, parallel_n_jobs)
            
            current_solutions, current_scores = self._accept_solutions(prev_solutions, prev_scores, 
                                                                       solution_array, step_scores, sa_temperature, optimization_type)
            sa_temperature = sa_temperature * cooling_rate

            new_pairs = parse_pairs(current_solutions, current_scores)
            example_pairs = new_pairs

            avg_step_score = sum(step_scores) / len(solution_array)
            best_score_per_step.append(best_step_score)
            avg_score_per_step.append(avg_step_score)
            best_score_history.append(best_score)
            if verbose > 0: 
                log_info(f"Step {step} - SA Temperature: {sa_temperature:.2f} - Current Best Score: {best_score:.3f}, Average Batch Score: {avg_step_score:.3f} - Best Batch Score: {best_step_score:.3f}")
            if verbose > 1:
                log_info(f"Best solution: {best_solution}")

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
            
            if sa_temperature < final_sa_temperature:
                log_warning(f"SA Temperature is too low. Stopping the optimization process.")
                break


        return OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            best_score_history=best_score_history,
            best_score_per_step=best_score_per_step,
            avg_score_per_step=avg_score_per_step
        )


