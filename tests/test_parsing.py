import pytest
from llmize.utils.parsing import parse_response, parse_pairs, parse_score

def test_parse_response_basic():
    """Test basic solution parsing without hyperparameters"""
    input_text = """
    <sol> 1,2,3 </sol>
    <sol> 4,5,6 </sol>
    """
    expected = [[1, 2, 3], [4, 5, 6]]
    result = parse_response(input_text)
    assert result == expected

def test_parse_response_with_hyperparameters():
    """Test parsing solutions with hyperparameters"""
    input_text = """
    <hp> 0.1, 0.8, 0.9 </hp>
    <sol> 1,2,3 </sol>
    <sol> 4,5,6 </sol>
    """
    expected_solutions = [[1, 2, 3], [4, 5, 6]]
    expected_hp = [0.1, 0.8, 0.9]
    solutions, hp = parse_response(input_text, hp_parse=True)
    assert solutions == expected_solutions
    assert hp == expected_hp

    

def test_parse_response_with_float_values():
    """Test parsing solutions with float values"""
    input_text = """
    <sol> 1.5, 2.7, 3.9 </sol>
    <sol> 4.2, 5.6, 6.1 </sol>
    """
    expected = [[1.5, 2.7, 3.9], [4.2, 5.6, 6.1]]
    result = parse_response(input_text)
    assert result == expected

def test_parse_response_with_negative_values():
    """Test parsing solutions with negative values"""
    input_text = """
    <sol> -1, -2.5, 3 </sol>
    <sol> 4, -5.6, -6 </sol>
    """
    expected = [[-1, -2.5, 3], [4, -5.6, -6]]
    result = parse_response(input_text)
    assert result == expected

def test_parse_response_invalid_input():
    """Test parsing with invalid input"""
    input_text = "Invalid input without sol tags"
    result = parse_response(input_text)
    assert result is None

def test_parse_pairs():
    """Test converting samples and scores to paired format"""
    samples = [[1, 2, 3], [4, 5, 6]]
    scores = [10.5, 20.7]
    expected = "\n<sol> 1,2,3 <\\sol>\nscore: 10.500\n\n<sol> 4,5,6 <\\sol>\nscore: 20.700\n"
    result = parse_pairs(samples, scores)
    assert result == expected

def test_parse_score():
    """Test parsing scores from text"""
    input_text = """
    <sol> 1,2,3 </sol>
    score: 10.50
    <sol> 4,5,6 </sol>
    score: 20.70
    """
    expected = [10.50, 20.70]
    result = parse_score(input_text)
    assert result == expected

if __name__ == "__main__":
    test_parse_score()

def test_parse_response_with_alternative_closing_tag():
    """Test parsing solutions with alternative closing tag format"""
    input_text = """
    <sol> 1,2,3 </sol>
    <sol> 4,5,6 <\\\\sol>
    """
    expected = [[1, 2, 3], [4, 5, 6]]
    result = parse_response(input_text)
    assert result == expected

def test_parse_response_with_extra_whitespace():
    """Test parsing solutions with extra whitespace"""
    input_text = """
    <sol>  1  ,  2  ,  3  </sol>
    <sol>  4  ,  5  ,  6  </sol>
    """
    expected = [[1, 2, 3], [4, 5, 6]]
    result = parse_response(input_text)
    assert result == expected

def test_parse_response_with_invalid_hp():
    """Test parsing with invalid hyperparameters"""
    input_text = """
    <hp> invalid, 0.8, 0.9 </hp>
    <sol> 1,2,3 </sol>
    """
    solutions, hp = parse_response(input_text, hp_parse=True)
    assert solutions == [[1, 2, 3]]
    assert hp == [0.8, 0.9]  # Skips invalid value and returns valid ones

def test_parse_pairs_empty_input():
    """Test parse_pairs with empty input"""
    samples = []
    scores = []
    expected = ""
    result = parse_pairs(samples, scores)
    assert result == expected

def test_parse_score_no_scores():
    """Test parse_score with no scores in input"""
    input_text = """
    <sol> 1,2,3 </sol>
    <sol> 4,5,6 </sol>
    """
    expected = []
    result = parse_score(input_text)
    assert result == expected