import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import multiprocessing

from cas5_functions import objective_function, template_content

from llmize import OPRO, HLMEA, HLMSA
from llmize.callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau
from llmize.utils.logger import log_info, log_error
from llmize.optimization_result import OptimizationResult


# Experiment parameters
#models = ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp"]
models = ["gemini-2.0-flash"]
#methods = [OPRO, HLMEA, HLMSA]
methods = [HLMEA, HLMSA]
num_trials = 5
batch_size = 16
num_steps = 250

# Generate initial solution by random perturbation of this:
# ["FUE1_enr", "FUE2_enr", "FUE3_enr", "FUE4_enr", "FUE5_enr", "FUE6_enr", "FUE7_enr",
# "FUE8_enr", "FUE8_gads", "FUE9_enr", "FUE9_gads", "FUE10_enr", "FUE10_gads", "FUE11_enr", "FUE11_gads"]

# enrichments are between 1 to 5% with 0.1% step
# gadolinia has maximum 10% with increment of 1% step

# set seed for reproducibility
np.random.seed(42)

def generate_initial_solutions(n_solutions=10):
    solutions = []
    for _ in range(n_solutions):
        # Generate random enrichments between 1-5% with 0.1% steps
        enrichments = np.round(np.random.choice(np.arange(3.6, 5.1, 0.1), size=11), decimals=1)
        enrichments[0] = np.round(np.random.choice(np.arange(1.5, 2.1, 0.1), size=1), decimals=1)
        enrichments[1] = np.round(np.random.choice(np.arange(2.0, 3.1, 0.1), size=1), decimals=1)
        enrichments[2] = np.round(np.random.choice(np.arange(2.0, 3.1, 0.1), size=1), decimals=1)

        # Generate random gadolinia content between 0-10% with 1% steps
        gads = np.round(np.random.choice(np.arange(4.0, 10.0, 1.0), size=4), decimals=1)
        # Combine into solution array
        solution = []
        for i in range(7):  # FUE1-7 enrichments only
            solution.append(enrichments[i])
        
        # Add FUE8-11 enrichments and gadolinia
        solution.extend([enrichments[7], gads[0]])  # FUE8
        solution.extend([enrichments[8], gads[1]])  # FUE9  
        solution.extend([enrichments[9], gads[2]])  # FUE10
        solution.extend([enrichments[10], gads[3]]) # FUE11
        
        solutions.append(solution)
        
    return solutions

solutions = generate_initial_solutions(n_solutions=batch_size)

def evaluate_solution(solution):
    score = objective_function(solution_list=solution, template_content=template_content)
    return score    

with multiprocessing.Pool() as pool:
    scores = pool.map(evaluate_solution, solutions)

# Callbacks
callbacks = [
    EarlyStopping(monitor='best_score', min_delta=0.01, patience=50, verbose=0),
    OptimalScoreStopping(optimal_score=100.0, tolerance=0.01), # Adjusted for this scale
    AdaptTempOnPlateau(monitor='best_score', init_temperature=1.0, min_delta=0.01, patience=20, factor=1.1, max_temperature=1.9, verbose=1)
]

with open("bwr_ge14.txt", "r") as f:
    problem_text = f.read()

def obj_func(x):
    return objective_function(x, template_content)

# Results storage
results = {}

# Run experiments
for model in models:
    for method in methods:
        method_name = method.__name__
        results[f"{model}_{method_name}"] = []
        
        log_info(f"\nRunning experiments for {method_name} with {model}")
        
        for trial in range(num_trials):
            log_info(f"\nTrial {trial + 1}/{num_trials}")
            
            optimizer = method(
                problem_text=problem_text,
                obj_func=obj_func,
                api_key=os.getenv("GEMINI_API_KEY"),
                llm_model=model
            )
            
            try:
                result = optimizer.maximize(
                    init_samples=solutions.copy(),
                    init_scores=scores.copy(),
                    batch_size=batch_size,
                    num_steps=num_steps,
                    callbacks=callbacks,
                    verbose=1,
                    parallel_n_jobs=-1
                )
            except Exception as e:
                log_error(f"Optimization failed: {str(e)}")
                result = OptimizationResult(
                    best_score=float('inf'),
                    best_solution=None,
                    best_score_history=[],
                    best_score_per_step=[],
                    avg_score_per_step=[]
                )
            
            results[f"{model}_{method_name}"].append({
                'best_score': result.best_score,
                'best_solution': result.best_solution,
                'best_score_history': result.best_score_history,
                'best_score_per_step': result.best_score_per_step,
                'avg_score_per_step': result.avg_score_per_step
            })
            
            log_info(f"Trial {trial + 1} best score: {result.best_score:.2f}")
            log_info(f"Trial {trial + 1} best solution: {result.best_solution}")

# Store results in current directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f'tsp_results_{time.strftime("%Y%m%d_%H%M")}.json')
with open(results_file, 'w') as f:
    json.dump(results, f)

log_info(f"Results saved to {results_file}")