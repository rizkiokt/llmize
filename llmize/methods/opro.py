import numpy as np

from ..base import Optimizer
from ..llm.llm_init import initialize_llm
from ..utils.parsing import parse_pairs, parse_response
from ..llm.llm_call import generate_content
from ..utils.decorators import check_init

from ..callbacks import EarlyStopping, AdaptTempOnPlateau

class OPRO(Optimizer):
    """
    OPRO optimizer for optimizing tasks using a specified LLM model.

    This class inherits from the `Optimizer` class and allows configuration 
    of various parameters related to the optimization process.

    :param str llm_model: The name of the LLM model to use (default: "gemini-2.0-flash").
    :param str api_key: The API key for accessing the model (default: None).
    :param int num_steps: The number of optimization steps (default: 50).
    :param int batch_size: The batch size used for optimization (default: 5).
    """

    def __init__(self, problem_text=None, obj_func=None, llm_model="gemini-2.0-flash", api_key=None):
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

Each solution should start with <sol> and end with <\sol> with a comma between parameters. 
Make sure the length of solutions match examples given. Don't guess for the scores as they will be calculated by an objective function.
"""
        prompt = "\n".join([self.problem_text, example_texts, example_pairs, instruction])

        return prompt

    def optimize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None, optimization_type="maximize"):
        """
        Perform the OPRO optimization process, either maximizing or minimizing the objective function.
        
        :param str problem_text: The textual description of the problem to be optimized.
        :param list init_samples: Initial solution samples provided as input (default: None).
        :param list init_scores: Scores corresponding to the initial samples (default: None).
        :param callable obj_func: Objective function to evaluate generated solutions (default: None).
        :param str optimization_type: Whether to maximize or minimize the objective function ('maximize' or 'minimize').
        
        :return: A dictionary containing the best solution, its score, and optimization history.
        :rtype: dict
        """
        # Initialize the LLM model
        client = initialize_llm(self.llm_model, self.api_key)

        print(f"Running OPRO optimization with {num_steps} steps and batch size {batch_size}...")
        best_solution = None
        if optimization_type == "maximize":
            best_score = np.max(init_scores)
        elif optimization_type == "minimize":
            best_score = np.min(init_scores)
        else:
            raise ValueError("Invalid optimization_type. Choose 'maximize' or 'minimize'.")
        
        best_score_history = [best_score]
        avg_score_per_step = [np.average(init_scores)]
        best_score_per_step = [best_score]

        max_retries = 5

        for step in range(num_steps+1):
            if step == 0:
                print(f"Step {step} - Best Initial Score: {best_score:.2f}, Average Initial Score: {np.average(init_scores):.2f}")
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
                continue

            prompt = self.meta_prompt(batch_size, example_pairs, optimization_type)

            response = generate_content(client, self.llm_model, prompt, temperature)
            solution_array = parse_response(response)

            retry = 0
            while solution_array is None or len(solution_array) != batch_size:
                print("Number of solutions parsed is not equal to batch size. Retrying...")
                response = generate_content(client, self.llm_model, prompt, temperature)
                solution_array = parse_response(response)

                retry += 1
                if retry >= max_retries:
                    raise ValueError("Failed to generate solutions after multiple attempts.")


            step_scores = []
            if optimization_type == "maximize":
                best_step_score = -float('inf')  # Start with the lowest possible value for maximization
            elif optimization_type == "minimize":
                best_step_score = float('inf')   # Start with the highest possible value for minimization

            for solution in solution_array:
                score = self.obj_func(solution)
                step_scores.append(score)

                if (optimization_type == "maximize" and score > best_step_score) or \
                (optimization_type == "minimize" and score < best_step_score):
                    best_step_score = score

                if (optimization_type == "maximize" and score > best_score) or \
                (optimization_type == "minimize" and score < best_score):
                    best_score = score
                    best_solution = solution

            new_pairs = parse_pairs(solution_array, step_scores)
            example_pairs = example_pairs + new_pairs

            avg_step_score = sum(step_scores) / len(solution_array)
            best_score_per_step.append(best_step_score)
            avg_score_per_step.append(avg_step_score)
            best_score_history.append(best_score)
            print(f"Step {step} - Current Best Score: {best_score:.2f}, Average Batch Score: {avg_step_score:.2f} - Best Batch Score: {best_step_score:.2f}")

            # Callbacks: Trigger at the end of each step
            if callbacks:
                for callback in callbacks:
                    logs = {callback.monitor: best_score}  # Pass logs with the monitored metric
                    new_temperature = callback.on_step_end(step, logs)  # Callback could adjust the temperature
                    if new_temperature is not None:
                        temperature = new_temperature  # Update temperature if needed
                    # Check if early stopping is triggered
                    if isinstance(callback, EarlyStopping) and callback.wait >= callback.patience:
                        print(f"Early stopping triggered at step {step}.")
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

    @check_init
    def maximize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None):
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature, 
                              callbacks=callbacks, optimization_type="maximize")
    
    @check_init
    def minimize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None):
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature,
                                callbacks=callbacks, optimization_type="minimize")

