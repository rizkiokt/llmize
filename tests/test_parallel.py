import pytest
import multiprocessing as mp
import os
import time
import numpy as np

from llmize import OPRO

def test_parallel_evaluation():
    """Test parallel evaluation of solutions in OPRO."""
    def obj_func(x):
        if isinstance(x, list):
            return (float(x[0]) + 2) ** 2  # Minimum at x=-2
        else:
            return (float(x) + 2) ** 2  # Minimum at x=-2
    
    opro = OPRO(
        problem_text="Minimize (x+2)^2",
        obj_func=obj_func, api_key=os.getenv("GEMINI_API_KEY")
    )
    
    init_samples = ["0", "1", "-1", "-3", "-2.5"]
    
    # Test sequential evaluation
    best_score_seq, best_sol_seq, step_scores_seq, best_step_score_seq = opro._evaluate_solutions(
        solution_array=init_samples,
        best_solution=None,
        optimization_type="minimize",
        verbose=0,
        best_score=None,
        parallel_n_jobs=1
    )
    
    # Test parallel evaluation with multiple cores
    best_score_par, best_sol_par, step_scores_par, best_step_score_par = opro._evaluate_solutions(
        solution_array=init_samples,
        best_solution=None,
        optimization_type="minimize",
        verbose=0,
        best_score=None,
        parallel_n_jobs=-1
    )
    
    # Ensure parallel and sequential results match
    assert best_score_seq == best_score_par, "Parallel and sequential evaluations should yield the same best score."
    assert best_sol_seq == best_sol_par, "Parallel and sequential evaluations should yield the same best solution."
    assert step_scores_seq == step_scores_par, "Parallel and sequential evaluations should yield the same step scores."
    assert best_step_score_seq == best_step_score_par, "Parallel and sequential evaluations should yield the same best step score."
    
    # Ensure best score is reasonable
    assert best_score_seq <= 0.25, "Best score should be close to zero, near the minimum at x=-2."

def obj_func(x):
    """Standalone objective function to ensure multiprocessing compatibility."""
    time.sleep(0.1)
    
    if isinstance(x, list):
        return sum((float(xi) + 2) ** 2 for xi in x)  # Minimum at x=-2 for all dimensions
    else:
        return (float(x) + 2) ** 2  # Minimum at x=-2

@pytest.mark.long_test
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Need GEMINI_API_KEY to run this test")
def test_time_parallel_evaluation():
    """Test parallel evaluation of solutions in OPRO with complex objective functions and time assertions."""
    opro = OPRO(
        problem_text="Minimize sum((x+2)^2)",
        obj_func=obj_func, api_key=os.getenv("GEMINI_API_KEY")
    )
    
    init_samples = [[str(i), str(i - 1)] for i in range(-10, 1)]  # Focus on values closer to -2
    
    # Measure sequential execution time
    start_seq = time.time()
    best_score_seq, best_sol_seq, step_scores_seq, best_step_score_seq = opro._evaluate_solutions(
        solution_array=init_samples,
        best_solution=None,
        optimization_type="minimize",
        verbose=0,
        best_score=None,
        parallel_n_jobs=1
    )
    end_seq = time.time()
    seq_time = end_seq - start_seq
    
    # Measure parallel execution time
    n_jobs = min(mp.cpu_count(), len(init_samples))
    start_par = time.time()
    best_score_par, best_sol_par, step_scores_par, best_step_score_par = opro._evaluate_solutions(
        solution_array=init_samples,
        best_solution=None,
        optimization_type="minimize",
        verbose=0,
        best_score=None,
        parallel_n_jobs=n_jobs
    )
    end_par = time.time()
    par_time = end_par - start_par
    
    # Ensure parallel and sequential results match
    assert best_score_seq == best_score_par, "Parallel and sequential evaluations should yield the same best score."
    assert best_sol_seq == best_sol_par, "Parallel and sequential evaluations should yield the same best solution."
    assert step_scores_seq == step_scores_par, "Parallel and sequential evaluations should yield the same step scores."
    assert best_step_score_seq == best_step_score_par, "Parallel and sequential evaluations should yield the same best step score."
    
    # Ensure best score is reasonable
    #assert best_score_seq <= 0.25, f"Best score should be close to zero, near the minimum at x=-2. Found: {best_score_seq}."

    # Assert that parallel execution is meaningfully faster than sequential execution
    scalability_factor = 2.0  # Allowing for parallel overhead
    assert par_time < (seq_time / n_jobs) * scalability_factor, \
        f"Parallel execution ({par_time}s) should be noticeably faster than sequential execution ({seq_time}s) considering {n_jobs} jobs."
