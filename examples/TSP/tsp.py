import numpy as np
from functools import partial
import os

from llmize import OPRO

def initialize_tsp(num_cities, seed=42):
    """
    Initializes the TSP with random city coordinates.
    :param num_cities: Number of cities
    :param seed: Random seed for reproducibility
    :return: Distance matrix
    """
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2) * 100  # Random cities in 100x100 space
    
    # Compute distance matrix
    dist_matrix = np.sqrt(((cities[:, np.newaxis] - cities[np.newaxis, :]) ** 2).sum(axis=2))
    return dist_matrix

def objective_function(route, dist_matrix):
    """
    Computes the total travel distance for a given route.
    :param route: A permutation of city indices representing the tour
    :param dist_matrix: Precomputed distance matrix
    :return: Total distance of the route
    """
    total_distance = sum(dist_matrix[route[i], route[i+1]] for i in range(len(route) - 1))
    total_distance += dist_matrix[route[-1], route[0]]  # Return to starting city
    return total_distance

# Example Usage
num_cities = 10
dist_matrix = initialize_tsp(num_cities)
# generate 5 random routes and total distances
routes = [np.random.permutation(num_cities) for _ in range(5)]
total_distances = [objective_function(route, dist_matrix) for route in routes]

# Initialize the OPRO optimizer
opro = OPRO(llm_model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

# change current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# run optimization
with open("tsp_problem.txt", "r") as f:
    problem_text = f.read()

results = opro.minimize(problem_text=problem_text, init_samples=routes, init_scores=total_distances,
                         obj_func=partial(objective_function, dist_matrix=dist_matrix))

print(results['best_solution'])
print(results['best_score'])

