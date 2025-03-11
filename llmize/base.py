import functools
from .utils.parsing import parse_pairs
from .llm.llm_call import generate_content
from .llm.llm_init import initialize_llm
from .utils.logger import log_info, log_critical, log_debug, log_warning, log_error
from .utils.parsing import parse_response
from .utils.decorators import check_init, time_it

class Optimizer:
    def __init__(self, problem_text=None, obj_func=None, llm_model="gemini-2.0-flash", api_key=None):
        """
        Initialize the Optimizer with the general configuration.
        """
        self.problem_text = problem_text
        self.obj_func = obj_func
        self.llm_model = llm_model
        self.api_key = api_key

    @check_init
    @time_it
    def maximize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None, verbose=1):
        """
        Run the optimization algorithm to maximize the objective function.

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
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature, 
                              callbacks=callbacks, verbose=verbose, optimization_type="maximize")
    
    @check_init
    @time_it
    def minimize(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None, verbose=1):
        """
        Run the optimization algorithm to minimize the objective function.

        Parameters:
        - init_samples (list): A list of initial solutions.
        - init_scores (list): A list of initial scores corresponding to init_samples.
        - num_steps (int): The number of optimization steps (default: 50).
        - batch_size (int): The number of new solutions to generate at each step (default: 5).
        - temperature (float): The temperature for the LLM model (default: 1.0).
        - callbacks (list): A list of callback functions to be triggered at the end of each step.
        - optimization_type (str): "maximize" or "minimize" (default: "minimize").

        Returns:
        - results (dict): A dictionary containing the best solution, best score, best score history, best score per step, and average score per step.
        """
        return self.optimize(init_samples=init_samples, init_scores=init_scores,
                              num_steps=num_steps, batch_size=batch_size, temperature=temperature,
                                callbacks=callbacks, verbose=verbose, optimization_type="minimize")
    
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

        """
        Generate a sample prompt for the language model based on the provided parameters.

        Parameters:
        - batch_size (int): The number of new solutions to generate (default: 5).
        - optimization_type (str): The type of optimization to perform, either "maximize" or "minimize" (default: "maximize").
        - init_samples (list): A list of initial solutions (default: None).
        - init_scores (list): A list of initial scores corresponding to init_samples (default: None).

        Returns:
        - str: The generated prompt as a string to be used for generating solutions from the language model.
        """
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

    def _generate_solutions(self, client, prompt, temperature, batch_size, verbose, max_retries=5):
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

        response = generate_content(client, self.llm_model, prompt, temperature)
        solution_array = parse_response(response)

        if verbose > 2: 
            #log_debug(f"Prompt: {prompt}")
            log_debug(f"Response: {response}")
        if verbose > 1: log_debug(f"Generated Solutions: {solution_array}")

        retry = 0
        
        while solution_array is None or len(solution_array) != batch_size:
            log_warning("Number of solutions parsed is not equal to batch size. Retrying...")
            response = generate_content(client, self.llm_model, prompt, temperature)
            solution_array = parse_response(response)

            if verbose > 2: 
                log_debug(f"Response for retry {retry+1}: {response}")
            if verbose > 1:
                log_debug(f"Generated Solutions for retry {retry+1}: {solution_array}")

            retry += 1
            if retry >= max_retries:
                log_critical("Failed to generate solutions after multiple attempts.")
                raise ValueError("Failed to generate solutions after multiple attempts.")

        return solution_array
    
    def _evaluate_solutions(self, solution_array, best_solution, optimization_type, verbose, best_score=None):
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
        
        for solution in solution_array:
            try:
                score = self.obj_func(solution)
            except Exception as e:
                log_error(f"Error occurred while evaluating solution {solution}: {e}")
                continue
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
    

