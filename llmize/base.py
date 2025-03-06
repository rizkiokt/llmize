import functools
from .utils.parsing import parse_pairs
from .llm.llm_call import generate_content
from .llm.llm_init import initialize_llm

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

        