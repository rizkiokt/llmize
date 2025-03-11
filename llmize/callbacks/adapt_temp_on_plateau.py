from ..utils.logger import log_info, log_debug

class AdaptTempOnPlateau:
    def __init__(self, monitor='best_score', init_temperature = 1.0, min_delta=0.0001, patience=10, factor=1.1, max_temperature=2.0, verbose=0):
        """
        Adapt temperature on plateau callback to increase the temperature when the metric plateaus.
        
        :param monitor: Metric to monitor (default: 'best_score').
        :param min_delta: Minimum change to qualify as an improvement (default: 0.0001).
        :param patience: Number of steps with no improvement before increasing temperature (default: 10).
        :param factor: Factor by which the temperature will be increased (default: 1.2).
        :param verbose: Verbosity mode. 0 = no output, 1 = print message when temperature is adapted (default: 0).
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self.wait = 0  # Counter for patience
        self.best_score = None  # Best score encountered so far
        self.temperature = init_temperature  # Initial temperature
        self.max_temperature = max_temperature
        self.stopped_step = 0  # Step when the temperature adaptation occurred

    def on_step_end(self, step, logs=None):
        """
        Check the stopping condition at the end of each step and adapt temperature if necessary.
        
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
            if self.verbose > 1:
                log_debug(f"No improvement in {self.monitor}. Patience count: {self.wait}/{self.patience}")
        else:
            self.best_score = current_score
            self.wait = 0  # Reset wait count

        # Adapt temperature if patience is exceeded
        if self.wait >= self.patience:
            self.temperature *= self.factor
            if self.temperature > self.max_temperature:
                self.temperature = self.max_temperature
            self.wait = 0  # Reset the wait counter
            self.stopped_step = step
            if self.verbose == 1:
                log_info(f"No improvement in {self.monitor} for {self.patience} steps. Adapted temperature to {self.temperature:.2f}.")
            return self.temperature  # Return the updated temperature

        return self.temperature
