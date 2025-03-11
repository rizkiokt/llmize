from ..utils.logger import log_info, log_warning
class EarlyStopping:
    def __init__(self, monitor='best_score', min_delta=0.01, patience=10, verbose=0):
        """
        Early stopping callback to monitor a specified metric and stop if no improvement.
        
        :param monitor: Metric to monitor (default: 'best_score').
        :param min_delta: Minimum change to qualify as an improvement (default: 0.01).
        :param patience: Number of steps with no improvement before stopping (default: 10).
        :param verbose: Verbosity mode. 0 = no output, 1 = print message when early stopping triggers (default: 0).
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0  # Counter for patience
        self.best_score = None  # Best score encountered so far
        self.stopped_step = 0  # Step when the optimization stopped

    def on_step_end(self, step, logs=None):
        """
        Check the stopping condition at the end of each step.
        
        :param step: The current step number.
        :param logs: Dictionary containing the logs (contains the metric to monitor).
        """
        current_score = logs.get(self.monitor)

        # If this is the first step, initialize best_score
        if self.best_score is None:
            self.best_score = current_score

        # Check if the current score has improved
        if abs(current_score - self.best_score) < self.min_delta:
            self.wait += 1
            if self.verbose > 0:
                log_info(f"No improvement in {self.monitor}. Patience count: {self.wait}/{self.patience}")
        else:
            self.best_score = current_score
            self.wait = 0  # Reset wait count

        # Check if early stopping should be triggered
        if self.wait >= self.patience:
            self.stopped_step = step
            log_warning(f"Early stopping triggered at step {step}.")
            return True  # Indicate that optimization should stop
        return False

