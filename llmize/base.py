class Optimizer:
    def __init__(self, llm_model="gemini-2.0-flash", api_key=None, num_steps=50, batch_size=5):
        """
        Initialize the Optimizer with the general configuration.
        """
        self.llm_model = llm_model
        self.api_key = api_key
        self.num_steps = num_steps
        self.batch_size = batch_size
    
    def optimize(self, problem_text=None, init_samples=None, init_scores=None, obj_func=None):
        """
        Placeholder method to be implemented by subclasses.
        """
        raise NotImplementedError("The optimize method must be implemented by the subclass.")
        
    def get_configuration(self):
        """
        Returns the general configuration of the optimizer.
        """
        return {
            "llm_model": self.llm_model,
            "api_key": self.api_key,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size
        }

