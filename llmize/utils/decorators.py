def check_init(func):
    def inner(self, init_samples=None, init_scores=None, num_steps=50, batch_size=5,
                 temperature=1.0, callbacks=None):

        if init_samples is None or init_scores is None:
            raise ValueError("init_samples and init_scores must be provided.")

        if num_steps <= 0 or not isinstance(num_steps, int):
            raise ValueError("num_steps must be a positive integer.")
        if batch_size <= 0 or not isinstance(batch_size, int):
            raise ValueError("batch_size must be a positive integer.")

        return func(self, init_samples, init_scores, num_steps, batch_size, temperature, callbacks)
    return inner
