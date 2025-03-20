import os
import numpy as np
from llmize import OPRO, HLMEA, HLMSA
from llmize.callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau
from llmize.utils.logger import log_info, log_error
import matplotlib.pyplot as plt
import json
import time


def objective_convex_penalty(x):
    """
    Objective function for the Convex Optimization problem with penalties.
    The function is minimized.
    """
    x1, x2 = x
    f = (x1 - 3)**2 + (x2 + 2)**2 + np.sin(x1 + x2) + 4
    
    # Constraint violations
    penalty = 0
    large_penalty = 100  # Large penalty value

    if x1 < 0 or x1 > 5:
        penalty += large_penalty
    if x2 < 0 or x2 > 5:
        penalty += large_penalty

    return f + penalty


# Experiment parameters
models = ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp"]
#models = ["gemini-2.0-flash"]
methods = [OPRO, HLMEA, HLMSA]
#methods = [HLMEA, HLMSA]
num_trials = 5
batch_size = 16
num_steps = 250

# Generate random solutions (list of lists) and scores (list of floats)
num_samples = int(np.sqrt(batch_size))
x1_range = np.linspace(0, 4.5, num_samples)
x2_range = np.linspace(0, 4.5, num_samples)
solutions = []
scores = []
for i in range(num_samples):
    for j in range(num_samples):
        solutions.append([x1_range[i], x2_range[j]])
        scores.append(objective_convex_penalty([x1_range[i], x2_range[j]]))

# Problem description
problem_text = "convex_problem.txt"

# Callbacks
callbacks = [
    EarlyStopping(monitor='best_score', min_delta=0.01, patience=50, verbose=0),
    OptimalScoreStopping(optimal_score=7.90, tolerance=0.01), # Adjusted for this scale
    AdaptTempOnPlateau(monitor='best_score', init_temperature=1.0, min_delta=0.01, patience=20, factor=1.1, max_temperature=1.9, verbose=1)
]

# Results storage
results = {}

# Run experiments
for method in methods:
    method_name = method.__name__
    for model in models:
        results[f"{model}_{method_name}"] = []
        
        log_info(f"\nRunning experiments for {method_name} with {model}")
        
        for trial in range(num_trials):
            log_info(f"\nTrial {trial + 1}/{num_trials}")
            
            optimizer = method(
                problem_text=problem_text,
                obj_func=objective_convex_penalty,
                api_key=os.getenv("GEMINI_API_KEY"),
                llm_model=model
            )
            
            try:
                result = optimizer.minimize(
                    init_samples=solutions.copy(),
                    init_scores=scores.copy(),
                    batch_size=batch_size,
                    num_steps=num_steps,
                    callbacks=callbacks,
                    verbose=1
                )
            except Exception as e:
                log_error(f"Optimization failed: {str(e)}")
                result = {
                    'best_score': float('inf'),
                    'best_solution': None,
                    'best_score_history': []
                }
            
            results[f"{model}_{method_name}"].append({
                'best_score': result['best_score'],
                'best_solution': result['best_solution'],
                'best_score_history': result['best_score_history']
            })
            
            log_info(f"Trial {trial + 1} best score: {result['best_score']:.2f}")
            log_info(f"Trial {trial + 1} best solution: {result['best_solution']}")

# Store results in current directory
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f'convex_results_{time.strftime("%Y%m%d_%H%M")}.json')
with open(results_file, 'w') as f:
    json.dump(results, f)

log_info(f"Results saved to {results_file}")    