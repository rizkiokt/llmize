import matplotlib.pyplot as plt
import numpy as np

def plot_scores(results, filename=None):
    """
    Plot optimization score progression.

    Parameters
    ----------
    results : OptimizationResult
        Output of :func:`llmize.OPRO.minimize` or :func:`llmize.OPRO.maximize`.
    filename : str, optional
        If not None, save the plot to the specified filename.

    Notes
    -----
    If `filename` is None, the plot is displayed using matplotlib's interactive
    backend. If `filename` is not None, the plot is saved to disk and the
    interactive backend is closed.

    """
    score_history = results.best_score_history
    best_score_per_step = results.best_score_per_step
    avg_score_per_step = results.avg_score_per_step
    
    steps = np.arange(len(score_history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, score_history, label="Current Best Score", linestyle="-", marker="o", markersize=4, alpha=0.7)
    plt.plot(steps, best_score_per_step, label="Best Score Per Step", linestyle="--", linewidth=2)
    plt.plot(steps, avg_score_per_step, label="Average Score Per Step", linestyle="-.", linewidth=2)
    
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Optimization Score Progression")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    if filename is not None: 
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return