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
            sol = sol.split("</sol>")[0].split("<\sol>")[0].strip()  # Handle both / and \ in closing tag
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
            # find line starts with <hp> and ends with <\hp>
            hp_text = ""        
            for line in response_text.splitlines():
                if line.startswith("<hp>"):
                    hp_text = line
                    break
            hp_text = hp_text.replace("<hp>", "").replace("<\hp>", "").replace("</hp>", "").strip()
            hp = []
            for x in hp_text.split(","):
                try:
                    hp.append(float(x.strip()))
                except ValueError:
                    try:
                        hp.append(float(x.split()[1]))  # Try splitting with space and taking the second number
                    except ValueError as e:
                        log_error(f"Error parsing hyperparameter value '{x}': {e}")
                        return solution_array, None
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

if __name__ == "__main__":
    text = """
<hp> 0.1, 0.8, 0.9 <\hp>
<sol> 5,2,7,3,0,8,9,6,1,4 <\sol>
<sol> 4,2,7,0,6,3,5,8,9,1 <\sol>
<sol> 4,8,1,3,0,5,2,9,7,6 <\sol>
<sol> 5,2,4,3,0,8,9,6,1,7 <\sol>
<sol> 5,2,7,3,1,8,9,6,0,4 <\sol>
<sol> 4,2,7,0,5,3,6,8,9,1 <\sol>
<sol> 5,2,7,0,6,3,8,9,1,4 <\sol>
<sol> 4,2,8,0,6,3,5,7,9,1 <\sol>
<sol> 5,2,7,3,0,8,1,6,9,4 <\sol>
<sol> 4,2,7,0,1,3,5,8,9,6 <\sol>
<sol> 5,2,7,3,0,6,9,8,1,4 <\sol>
<sol> 4,2,7,1,6,3,5,8,9,0 <\sol>
<sol> 5,2,6,3,0,8,9,7,1,4 <\sol>
<sol> 4,7,2,0,6,3,5,8,9,1 <\sol>
<sol> 5,7,2,3,0,8,9,6,1,4 <\sol>
<sol> 4,2,7,0,6,3,1,8,9,5 <\sol>
    """
    sol, hp = parse_response(text, hp_parse=True)
    print(sol)
    print(hp)
