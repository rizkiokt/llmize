def parse_response(response_text):
    """
    Parses the generated response into a list of lists.
    Returns None if parsing fails.
    """
    try:
        solution_array  = []
        solutions = response_text.split("<sol>")[1:]  # Split solutions correctly
        for sol in solutions:
            sol = sol.split("</sol>")[0].strip()  # Fixed closing tag
            values = [float(x.strip()) for x in sol.split(",") if x.strip().replace('.', '', 1).replace('-', '', 1).isdigit()]
            solution_array .append(values)  # Store parsed values

        return solution_array   # Return the parsed solutions
    except Exception as e:
        print(f"Error: Failed to parse response due to unexpected error - {e}")
        return None

def parse_pairs(init_samples, init_scores):

    output = ""
    for sample, score in zip(init_samples, init_scores):
        output += f"<sol> {','.join(map(str, sample))} <\\sol>\nscore: {score}\n\n"

    return output.strip()

