import numpy as np

from ..base import Optimizer
from ..utils.llm_init import initialize_gemini, initialize_huggingface
from ..utils.parsing import parse_pairs, parse_response
from ..utils.llm_call import generate_content_gemini, generate_content_huggingface


class OPRO(Optimizer):
    def __init__(self, llm_model="gemini-2.0-flash", api_key=None, num_steps=50, batch_size=5):
        """
        Initialize the OPRO optimizer with the provided configuration.
        Inherits from Optimizer.
        """
        super().__init__(llm_model=llm_model, api_key=api_key, num_steps=num_steps, batch_size=batch_size)
    
    def meta_prompt(self, problem_text, example_pairs):

        instruction = f"""
Generate exactly {self.batch_size} new solutions that:
- Are distinct from all previous solutions.
- Have a higher score than the highest provided.
- Respect the relationships based on logical reasoning.

The solutions should start with <sol> and end with <\sol> with a comma between parameters. 
"""
        prompt = [problem_text, example_pairs, instruction]

        return prompt
    
    def optimize(self, problem_text=None, init_samples=None, init_scores=None, obj_func=None):
        """
        Perform the optimization process (unique to OPRO).
        Accepts problem-specific inputs.
        """
        if problem_text is None:
            raise ValueError("problem_text must be provided.")
        if init_samples is None or init_scores is None:
            raise ValueError("init_samples and init_scores must be provided.")
        if obj_func is None:
            raise ValueError("obj_func must be provided.")
        
        # Initialize the LLM model
        if self.llm_model.startswith("gemini"):
            client = initialize_gemini(self.api_key)
        else:
            client = initialize_huggingface(self.api_key)

        print(f"Running OPRO optimization with {self.num_steps} steps and batch size {self.batch_size}...")
        
        for step in range(self.num_steps):
            if step == 0: 
                init_pairs = parse_pairs(init_samples, init_scores)
                example_pairs = init_pairs
            prompt = self.meta_prompt(problem_text, example_pairs)

            if self.llm_model.startswith("gemini"):
                response = generate_content_gemini(client, self.llm_model, prompt)
            else:
                response = generate_content_huggingface(client, self.llm_model, prompt)
            
            if response is None:
                raise ValueError("LLM request failed. Please check your API key and try again.")

            solution_array = parse_response(response)

            while solution_array is None or len(solution_array) != self.batch_size:
                print("Number of solutions parsed is not equal to batch size. Retrying...")
                response = generate_content_huggingface(client, self.llm_model, prompt)
                solution_array = parse_response(response)
            
            step_scores = []

            for solution in solution_array:
                step_scores.append(obj_func(solution))
            
            new_pairs = parse_pairs(solution_array, step_scores)
            example_pairs = example_pairs + new_pairs

            step_scores = np.array(step_scores)
            best_index = np.argmax(step_scores)
            best_solution = solution_array[best_index]
            best_score = step_scores[best_index]
            print(f"Step {step + 1}: Best solution found: {best_solution}, score: {best_score}")


        # Example optimization logic (replace with actual logic)
        results = {
            "best_solution": "solution_xyz",
            "best_score": np.sum(init_scores),  # Placeholder logic
        }
        return results