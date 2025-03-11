import numpy as np

from ..base import Optimizer
from ..llm.llm_init import initialize_llm
from ..utils.parsing import parse_pairs
from ..utils.truncate import truncate_pairs
from ..utils.logger import log_info, log_warning, log_error, log_critical, log_debug
from ..callbacks import EarlyStopping

class HLMEA(Optimizer):
    """
    HLMEA: Hyper-heuristic LLM-driven Evolutionary Algorithm
    HLMEA optimizer for optimizing tasks using a specified LLM model.

    This class inherits from the `Optimizer` class and allows configuration 
    of various parameters related to the optimization process.

    :param str llm_model: The name of the LLM model to use (default: "gemini-2.0-flash").
    :param str api_key: The API key for accessing the model (default: None).
    :param int num_steps: The number of optimization steps (default: 50).
    :param int batch_size: The batch size used for optimization (default: 5).
    """

    def __init__(self, problem_text=None, obj_func=None, llm_model="gemini-2.0-flash", api_key=None):
        """
        Initialize the HLMEA optimizer with the provided configuration.
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

        backstory = """
You are a hyper-heuristic LLM-driven evolutionary algorithm that generates new solutions for a given problem.
You are capable of selecting the most optimal hyperparameters of evolutionary algorithms based on your knowledge and the problem at hand.
The problem is described as follows:
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

Directly give me the solutions in the format: <sol> param1, param2, ..., paramn <\sol> with a comma between parameters.
Don't give me any explanation, just the solutions and your decision on hyperparameters.
Make sure the length of solutions match examples given. Don't guess for the scores as they will be calculated by an objective function.
"""
        prompt = "\n".join([backstory, self.problem_text, example_texts, example_pairs, instruction])

        return prompt
    def optimize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None, verbose=1, optimization_type="maximize"):
        
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
        - results (dict): A dictionary containing the best solution, best score, best score history, best score per step, and average score per step.
        """

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
            raise ValueError("Invalid optimization_type. Choose 'maximize' or 'minimize'.")
        
        best_score_history = [best_score]
        avg_score_per_step = [np.average(init_scores)]
        best_score_per_step = [best_score]

        for step in range(num_steps+1):
            if step == 0:
                if verbose > 0: 
                    log_info(f"Step {step} - Best Initial Score: {best_score:.2f}, Average Initial Score: {np.average(init_scores):.2f}")
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
                continue
            
            if verbose > 1: 
                log_debug(f"Example pairs: {example_pairs}")

            prompt = self.meta_prompt(batch_size, example_pairs, optimization_type)


            solution_array = self._generate_solutions(client, prompt, temperature, 
                                                            batch_size, verbose)
            
            best_score, best_solution, step_scores, best_step_score = self._evaluate_solutions(solution_array, best_solution,
                                                                              optimization_type, verbose, best_score)
            new_pairs = parse_pairs(solution_array, step_scores)
            example_pairs = new_pairs

            avg_step_score = sum(step_scores) / len(solution_array)
            best_score_per_step.append(best_step_score)
            avg_score_per_step.append(avg_step_score)
            best_score_history.append(best_score)
            if verbose > 0: 
                log_info(f"Step {step} - Current Best Score: {best_score:.2f}, Average Batch Score: {avg_step_score:.2f} - Best Batch Score: {best_step_score:.2f}")

            # Callbacks: Trigger at the end of each step
            if callbacks:
                for callback in callbacks:
                    logs = {callback.monitor: best_score}  # Pass logs with the monitored metric
                    new_temperature = callback.on_step_end(step, logs)  # Callback could adjust the temperature
                    if new_temperature is not None:
                        temperature = new_temperature  # Update temperature if needed
                    # Check if early stopping is triggered
                    if isinstance(callback, EarlyStopping) and callback.wait >= callback.patience:
                        return {
                            "best_solution": best_solution,
                            "best_score": best_score,
                            "best_score_history": best_score_history,
                            "best_score_per_step": best_score_per_step,
                            "avg_score_per_step": avg_score_per_step
                        }

        results = {
            "best_solution": best_solution,
            "best_score": best_score,
            "best_score_history": best_score_history,
            "best_score_per_step": best_score_per_step,
            "avg_score_per_step": avg_score_per_step
        }

        return results


