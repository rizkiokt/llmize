import numpy as np

from ..base import Optimizer, OptimizationResult
from ..llm.llm_init import initialize_llm
from ..utils.parsing import parse_pairs
from ..utils.truncate import truncate_pairs
from ..utils.logger import log_info, log_warning, log_error, log_critical, log_debug
from ..callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau

class HLMEA(Optimizer):
    """
    :no-index:
    HLMEA optimizer for optimizing tasks using a specified LLM model.

    This class inherits from the `Optimizer` class and allows configuration 
    of various parameters related to the optimization process.

    :param str llm_model: The name of the LLM model to use (default from config).
    :param str api_key: The API key for accessing the model (default: None).
    :param int num_steps: The number of optimization steps (default: 50).
    :param int batch_size: The batch size used for optimization (default: 5).
    """

    def __init__(self, problem_text=None, obj_func=None, llm_model=None, api_key=None):
        """
        Initialize the HLMEA optimizer with the provided configuration.
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
        Generate a prompt for the LLM model to generate new solutions.
        Parameters:
        - batch_size (int): Number of new solutions to generate.
        - example_pairs (str): Example solutions and scores.
        - optimization_type (str): "maximize" or "minimize" (default: "maximize").

        Returns:
        - text: A formatted prompt structure.
        """

        backstory = """
You are a hyper-heuristic LLM-driven evolutionary algorithm that generates diverse and optimized solutions for a given problem.  
You can adaptively set hyperparameters to enhance solution quality and exploration.  

## Problem Description  
You need to optimize a given problem by evolving a set of candidate solutions. The objective is to iteratively improve solutions using evolutionary strategies, ensuring diversity and avoiding premature convergence.  
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

Generate **exactly {batch_size} diverse solutions** for the next generation using an evolutionary algorithm.  

### **Step 1: Adaptive Hyperparameter Selection**  
- Determine the **elitism rate, mutation rate, and crossover rate**.  
- Prioritize **high mutation rate and high crossover rate** to enhance diversity and avoid premature convergence.  

### **Step 2: Selection (Parent Selection)**  
- Select the best solutions from the previous population using **Roulette Wheel Selection, Tournament Selection, or Rank-Based Selection**.  
- Ensure that diverse parents are chosen to prevent premature convergence.  

### **Step 3: Crossover (Recombination of Parents)**  
- Generate new solutions using appropriate crossover techniques such as **Ordered Crossover (OX), Partially Mapped Crossover (PMX), or problem-specific recombination operators**.  
- Apply crossover **only if it introduces diversity**.  

### **Step 4: Mutation (Introducing Variations)**  
- Apply mutation based on the mutation rate. Mutation operators may include:  
  - **Swap Mutation**: Swap two random elements.  
  - **Insert Mutation**: Insert an element at a different position.  
  - **Inversion Mutation**: Reverse a segment of the solution.  
  - **Problem-Specific Mutations**: Apply domain-relevant perturbations.  
- Ensure each generated solution is **unique** compared to existing ones.  

### **Step 5: Uniqueness Enforcement**  
- **Check if a newly generated solution is identical to any previous or current solutions**.  
- If duplication occurs, **repeat Steps 3-5 until a new unique solution is found**.  


## **Output Format**  

Return exactly **{batch_size} unique solutions** and the chosen hyperparameters in the following format:  

### **Hyperparameters Output**  

<hp> elitism_rate, mutation_rate, crossover_rate <\\\\hp>

### **Solutions Output**  

<sol> param1, param2, ..., paramn <\\\\sol>

**Only provide the solutions and hyperparameters—do not include any extra text. Do not include any code.**  

"""
        
        instruction_full = f"""
The best solution has the {text2} score from the provided population.

Generate exactly {batch_size} solutions for the next population by following the step-by-step instuctions below.
1. Set the elitism rate, mutation rate, and crossover rate based on your knowledge. 
   Prioritize the hyperparameters that introduce diversity in the search space (high mutation rate and high crossover rate).
2. Select the best solutions to keep based on the elitism rate you decided.
3. Implement natural selection using strategies like Roulette Wheel Selection, Tournament Selection, or Rank-Based Selection for the selected mechanism.
4. Crossover the selected solutions and generate a new solution based on the crossover rate you decided. 
   There are 2 different crossover operators you can use: PMX (Partially Mapped Crossover), OX (Ordered Crossover).
5. Mutate the solution generated and generate a new solution based on the mutation rate you decided.
   There are 3 different mutation operators you can use: Swap Mutation, Insert Mutation, Inversion Mutation
6. If the solution generated is identical with one of the previous or current solutions, repeat the step 3-5.
7. Keep the solutions generated in step 2 and 6 and repeat step 3-6 until you generate {batch_size} solutions.
8. Think step by step to follow the instructions above. The format <sol> and <hp> are only used for your final output, don't include them in your process.


Give me the solutions in the format: <sol> param1, param2, ..., paramn <\\\\sol> with a comma between parameters.
Also, give me your decision on the hyperparameters in the format: <hp> elitism_rate, mutation_rate, crossover_rate <\\\\hp>.
Make sure the length of solutions match examples given. Don't guess for the scores as they will be calculated by an objective function.

"""
        prompt = "\n".join([backstory, self.problem_text, example_texts, hp_text, example_pairs, instruction])

        return prompt
    
    def optimize(self, init_samples=None, init_scores=None, num_steps=None, batch_size=None,
                 temperature=None, callbacks=None, verbose=1, optimization_type="maximize", parallel_n_jobs=None):
        
        """
        Run the HLMEA optimization algorithm.

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
            log_info(f"Running HLMEA optimization with {num_steps} steps and batch size {batch_size}...")
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

        max_examples = batch_size

        # Call the helper function to initialize callbacks
        self._initialize_callbacks(callbacks, temperature)

        for step in range(num_steps+1):
            if step == 0:
                if verbose > 0: 
                    log_info(f"Step {step} - Best Initial Score: {best_score:.3f}, Average Initial Score: {np.average(init_scores):.3f}")
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
                hp_text = "The solutions below are generated randomly."
                continue
            
            if verbose > 1: 
                log_debug(f"Example pairs: {example_pairs}")    

            prompt = self.meta_prompt(batch_size, example_pairs, optimization_type, hp_text)
            if verbose > 3:
                log_debug(f"Prompt: {prompt}")

            solution_array = self._generate_solutions(client, prompt, temperature, 
                                                            batch_size, verbose)
            
            best_score, best_solution, step_scores, best_step_score = self._evaluate_solutions(solution_array, best_solution,
                                                                              optimization_type, verbose, best_score, parallel_n_jobs)
            new_pairs = parse_pairs(solution_array, step_scores)
            example_pairs = example_pairs + new_pairs
            example_pairs = truncate_pairs(example_pairs, max_examples, optimization_type)

            avg_step_score = sum(step_scores) / len(solution_array)
            best_score_per_step.append(best_step_score)
            avg_score_per_step.append(avg_step_score)
            best_score_history.append(best_score)
            if verbose > 0: 
                log_info(f"Step {step} - Current Best Score: {best_score:.3f}, Average Batch Score: {avg_step_score:.3f} - Best Batch Score: {best_step_score:.3f}")
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

        return OptimizationResult(
            best_solution=best_solution,
            best_score=best_score,
            best_score_history=best_score_history,
            best_score_per_step=best_score_per_step,
            avg_score_per_step=avg_score_per_step
        )


