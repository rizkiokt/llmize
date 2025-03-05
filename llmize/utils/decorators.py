def check_init(func):
    def inner(self, problem_text=None, init_samples=None, init_scores=None, obj_func=None):
        if problem_text is None:
            raise ValueError("problem_text must be provided.")
        if init_samples is None or init_scores is None:
            raise ValueError("init_samples and init_scores must be provided.")
        if obj_func is None:
            raise ValueError("obj_func must be provided.")
        return func(self, problem_text, init_samples, init_scores, obj_func)
    return inner