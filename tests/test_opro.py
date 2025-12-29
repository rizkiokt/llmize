import pytest
import numpy as np
from llmize import OPRO
from llmize.callbacks import OptimalScoreStopping
from llmize.config import get_config
import os

def test_opro_initialization():
    """Test OPRO optimizer initialization"""
    problem_text = "Maximize x^2 where x is integer between -10 and 10"
    obj_func = lambda x: float(x)**2
    
    opro = OPRO(problem_text=problem_text, obj_func=obj_func, api_key=os.getenv("GEMINI_API_KEY"))
    
    assert opro.problem_text == problem_text
    assert opro.obj_func == obj_func
    assert opro.llm_model == get_config().default_model  # default model

@pytest.mark.very_long_test
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Need GEMINI_API_KEY to run this test")
def test_optimal_score_stopping():
    """Test OptimalScoreStopping callback functionality"""
    # Create a simple optimization problem
    def obj_func(x):
        if isinstance(x, list):
            return float(x[0])**2  # Simple quadratic function
        else:
            return float(x)**2  # Simple quadratic function
    
    opro = OPRO(
        problem_text="Maximize x^2 where x is integer between -10 and 10 (not including 10 and -10)",
        obj_func=obj_func, api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Initialize with target score of 81 (which would be achieved by x=9 or x=-9)
    optimal_stopping = OptimalScoreStopping(optimal_score=81, tolerance=0.1)
    
    # Test optimization with the callback
    init_samples = ["5", "-4", "3"]  # String inputs as that's what LLM would generate
    init_scores = [25, 16, 9]  # Corresponding scores: 5^2, (-4)^2, 3^2
    
    result = opro.maximize(
        init_samples=init_samples,
        init_scores=init_scores,
        num_steps=50,
        batch_size=3,
        callbacks=[optimal_stopping]
    )
    
    # Check if optimization results contain expected fields
    assert hasattr(result, 'best_score')
    assert hasattr(result, 'best_solution')
    assert hasattr(result, 'best_score_history')
    assert isinstance(result.best_score_history, list)

@pytest.mark.long_test
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Need GEMINI_API_KEY to run this test")
def test_opro_minimize():
    """Test OPRO's minimize functionality"""
    def obj_func(x):
        if isinstance(x, list):
            return (float(x[0]) + 2)**2  # Minimum at x=-2
        else:
            return (float(x) + 2)**2  # Minimum at x=-2
    
    opro = OPRO(
        problem_text="Minimize (x+2)^2",
        obj_func=obj_func, api_key=os.getenv("GEMINI_API_KEY")
    )
    
    init_samples = ["0", "1", "-1"]
    init_scores = [4, 9, 1]  # (0+2)^2, (1+2)^2, (-1+2)^2
    
    result = opro.minimize(
        init_samples=init_samples,
        init_scores=init_scores,
        num_steps=2,
        batch_size=2
    )
    
    assert result.best_score <= 4  # Should be better than initial best score

def test_opro_with_invalid_optimization_type():
    """Test OPRO with invalid optimization type"""
    opro = OPRO(
        problem_text="Test problem",
        obj_func=lambda x: float(x), api_key=os.getenv("GEMINI_API_KEY")
    )
    
    with pytest.raises(ValueError, match="optimization_type must be 'maximize' or 'minimize'"):
        opro.optimize(
            init_samples=["1"],
            init_scores=[1],
            optimization_type="invalid"
        )