from ..callbacks import EarlyStopping
from ..utils.logger import log_warning

class OptimalScoreStopping(EarlyStopping):
    def __init__(self, optimal_score, tolerance=0.01):
        super().__init__(monitor='best_score')
        self.optimal_score = optimal_score
        self.tolerance = tolerance
    
    def on_step_end(self, step, logs=None):
        current_score = logs.get(self.monitor)
        if abs(current_score - self.optimal_score) <= self.tolerance:
            log_warning(f"Optimal score reached at step {step}.")
            return True
        return False