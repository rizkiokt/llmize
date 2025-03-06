import numpy as np

from llmize.utils.parsing import parse_response, parse_pairs, parse_score

def truncate_pairs(pairs, nmax=10, optimization_type="maximize", method="mixed"):

    """
    Truncate a list of pairs to a maximum number of solutions.

    Parameters:
    - pairs (str): A string of example solutions and scores.
    - nmax (int): The maximum number of solutions to keep. Default is 10.
    - optimization_type (str): "maximize" or "minimize". Default is "maximize".
    - method (str): "best", "random", or "mixed". Default is "mixed".

    Returns:
    - truncated_pairs (str): A string of the truncated list of pairs.

    Notes:
    - The "best" method keeps the top nmax solutions based on the scores.
    - The "random" method randomly selects nmax solutions from the list.
    - The "mixed" method keeps the top nmax//2 solutions and randomly selects nmax-nmax//2 solutions from the remaining list.
    """
    solution_array = parse_response(pairs)
    scores = parse_score(pairs)

    if len(scores) <= nmax:
        return pairs

    if optimization_type == "maximize":
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    elif optimization_type == "minimize":
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
    random_indices = np.random.choice(sorted_indices, size=nmax, replace=False)

    if method=="best":
        truncated_sols = [solution_array[i] for i in sorted_indices[:nmax]]
        truncated_scores = [scores[i] for i in sorted_indices[:nmax]]
    elif method=="random":
        #randomize the sorted indices
        truncated_sols = [solution_array[i] for i in random_indices[:nmax]]
        truncated_scores = [scores[i] for i in random_indices[:nmax]]
    elif method=="mixed":
        # Mix the best nmax//2 with random nmax//2
        random_halves = np.random.choice(sorted_indices[nmax//2:], size=nmax-nmax//2, replace=False)
        truncated_sols = [solution_array[i] for i in sorted_indices[:nmax//2]] + [solution_array[i] for i in random_halves]
        truncated_scores = [scores[i] for i in sorted_indices[:nmax//2]] + [scores[i] for i in random_halves]
    
    truncated_pairs = parse_pairs(truncated_sols, truncated_scores)

    return truncated_pairs