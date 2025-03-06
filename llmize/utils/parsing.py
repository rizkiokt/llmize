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

        return solution_array   # Return the parsed solutions
    except Exception as e:
        print(f"Error: Failed to parse response due to unexpected error - {e}")
        return None


def parse_pairs(samples, scores):

    output = ""
    for sample, score in zip(samples, scores):
        output += f"<sol> {','.join(map(str, sample))} <\\sol>\nscore: {score:.2f}\n\n"

    return output.strip()

