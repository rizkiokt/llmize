import os
import numpy as np
from llmize import OPRO, HLMEA, HLMSA
from llmize.callbacks import EarlyStopping, OptimalScoreStopping, AdaptTempOnPlateau
from llmize.utils.logger import log_info, log_error
import matplotlib.pyplot as plt
import json
import time


def initialize_tsp(num_cities, seed=42):
    """
    Initializes the TSP with random city coordinates.
    """
    np.random.seed(seed)
    cities = (np.random.rand(num_cities, 2) * 100).astype(int)  # Random cities in 100x100 space
    dist_matrix = np.sqrt(((cities[:, np.newaxis] - cities[np.newaxis, :]) ** 2).sum(axis=2))
    return cities, dist_matrix

def objective_function(route, dist_matrix):
    """
    Computes the total travel distance for a given route.
    :param route: A permutation of city indices representing the tour
    :param dist_matrix: Precomputed distance matrix
    :return: Total distance of the route
    """

    # Check if route is valid
    if len(route) != dist_matrix.shape[0]:
        return 1000

    total_distance = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route) - 1))
    total_distance += dist_matrix[route[-1], route[0]]  # Return to starting city
    return total_distance


# Experiment parameters
#models = ["gemini-2.0-flash", "gemini-2.0-flash-thinking-exp"]
models = ["gemini-2.0-flash"]
#methods = [OPRO, HLMEA, HLMSA]
methods = [HLMEA, HLMSA]
num_trials = 5
batch_size = 16
num_steps = 250

# Initialize TSP problem
num_cities = 10
cities, dist_matrix = initialize_tsp(num_cities)
obj_func = lambda x: objective_function(x, dist_matrix)

# Problem description
problem_text = "tsp_problem.txt"

# Initial samples
init_samples = [np.random.permutation(num_cities) for _ in range(batch_size)]
init_scores = [objective_function(route, dist_matrix) for route in init_samples]

# Callbacks
callbacks = [
    EarlyStopping(monitor='best_score', min_delta=1.0, patience=50, verbose=0),
    OptimalScoreStopping(optimal_score=290.22, tolerance=0.01), # Adjusted for this scale
    AdaptTempOnPlateau(monitor='best_score', init_temperature=1.0, min_delta=1.0, patience=20, factor=1.1, max_temperature=1.9, verbose=1)
]

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
                result = optimizer.minimize(
                    init_samples=init_samples.copy(),
                    init_scores=init_scores.copy(),
                    batch_size=batch_size,
                    num_steps=num_steps,
                    callbacks=callbacks,
                    verbose=1
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