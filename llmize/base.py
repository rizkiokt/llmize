import functools
from .utils.parsing import parse_pairs
from .llm.llm_call import generate_content
from .llm.llm_init import initialize_llm
from .utils.logger import log_info, log_critical, log_debug, log_warning
from .utils.parsing import parse_response

class Optimizer:
    def __init__(self, problem_text=None, obj_func=None, llm_model="gemini-2.0-flash", api_key=None):
        """
        Initialize the Optimizer with the general configuration.
        """
        self.problem_text = problem_text
        self.obj_func = obj_func
        self.llm_model = llm_model
        self.api_key = api_key
    
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
    
    
    def get_sample_prompt(self, batch_size=5, optimization_type="maximize", init_samples=None, init_scores=None):

        example_pairs = parse_pairs(init_samples, init_scores)

        prompt = self.meta_prompt(batch_size=batch_size, example_pairs=example_pairs, optimization_type=optimization_type)

        return prompt

    def get_sample_response(self, prompt):

        client = initialize_llm(self.llm_model, self.api_key)

        return generate_content(client, self.llm_model, prompt)

    def _retry_generate_solutions(client, llm_model, prompt, temperature, batch_size, verbose, max_retries=5):
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
        
        Returns:
            A list of solutions that matches the expected batch_size.
        
        Raises:
            ValueError: If the expected number of solutions cannot be generated after max_retries.
        """
        retry = 0
        solution_array = None
        
        while solution_array is None or len(solution_array) != batch_size:
            log_warning("Number of solutions parsed is not equal to batch size. Retrying...")
            response = generate_content(client, llm_model, prompt, temperature)
            solution_array = parse_response(response)

            if verbose > 2: 
                log_debug(f"Response for retry {retry+1}:", response)
            if verbose > 1:
                log_debug(f"Generated Solutions for retry {retry+1}:", solution_array)

            retry += 1
            if retry >= max_retries:
                log_critical("Failed to generate solutions after multiple attempts.")
                raise ValueError("Failed to generate solutions after multiple attempts.")

        return solution_array
    
    def _evaluate_solutions(solution_array, optimization_type, verbose, obj_func, best_score=None):
        """
        Evaluate a list of solutions and update the best solution based on an objective function.
        
        Parameters:
            solution_array (list): A list of solutions to evaluate.
            optimization_type (str): "maximize" or "minimize".
            verbose (int): Verbosity level for logging.
            obj_func (callable): A function that takes a solution and returns its score.
            best_score (float, optional): The current best score. If not provided, it will be
                initialized based on the optimization_type.
                
        Returns:
            tuple: (best_score, best_solution, step_scores)
                best_score (float): The updated best score after evaluating solutions.
                best_solution (any): The solution corresponding to the best score.
                step_scores (list): A list of scores for each solution in solution_array.
        
        Raises:
            ValueError: If optimization_type is not "maximize" or "minimize".
        """
        step_scores = []
        
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
        
        best_solution = None

        for solution in solution_array:
            score = obj_func(solution)
            step_scores.append(score)

            if verbose > 1:
                log_debug(f"Score for solution {solution}: {score}")

            # Update best step score for the current batch
            if (optimization_type == "maximize" and score > best_step_score) or \
            (optimization_type == "minimize" and score < best_step_score):
                best_step_score = score

            # Update the overall best score and solution if found
            if (optimization_type == "maximize" and score > best_score) or \
            (optimization_type == "minimize" and score < best_score):
                best_score = score
                best_solution = solution

        return best_score, best_solution, step_scores, best_step_score
        