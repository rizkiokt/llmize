from llmize.utils.logger import log_info, log_error

def parse_response(response_text, hp_parse=False):
    """
    Parses the generated response into a list of lists.
    Returns None if parsing fails.
    """
    try:
        solution_array  = []
        solutions = response_text.split("<sol>")[1:]  # Split solutions correctly
        for sol in solutions:
            sol = sol.split("<\sol>")[0].strip()  # Fixed closing tag
            values = []
            for x in sol.split(","):
                x = x.strip()
                x = x.split()[0]
                if x.replace('.', '', 1).replace('-', '', 1).isdigit():  # Check if it's a number
                    try:
                        value = int(x)
                    except ValueError:
                        value = float(x)
                    values.append(value)
            solution_array.append(values)  # Store parsed values
        
        if hp_parse:
            hp = response_text.split("<hp>")[1:]
            hp = hp.split("<\hp>")[0].strip()
            hp = hp.split(",")
            hp = [float(x) for x in hp]
            return solution_array, hp

        return solution_array   # Return the parsed solutions
    except Exception as e:
        log_error(f"Error: Failed to parse response due to unexpected error - {e}")
        return None


def parse_pairs(samples, scores):

    output = ""
    for sample, score in zip(samples, scores):
        output += "\n"
        output += f"<sol> {','.join(map(str, sample))} <\\sol>\nscore: {score:.2f}\n"

    return output

def parse_score(text):

    scores = []
    for line in text.splitlines():
        if line.startswith("score:"):
            scores.append(float(line.split()[1]))


    return scores

