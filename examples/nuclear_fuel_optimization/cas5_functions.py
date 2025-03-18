import numpy as np
import os
import subprocess
import time
import re
import random

KEYS = ["FUE1_enr", "FUE2_enr", "FUE3_enr", "FUE4_enr", "FUE5_enr", "FUE6_enr", "FUE7_enr",
                    "FUE8_enr", "FUE8_gads", "FUE9_enr", "FUE9_gads", "FUE10_enr", "FUE10_gads", "FUE11_enr", "FUE11_gads"]
TEMPLATE_FILE = "GE14.DOM.BOC.inp"
FOLDER = "cas5_runs"

# check if folder exists
if os.path.exists(FOLDER):
    # remove all files in folder
    for file in os.listdir(FOLDER):
        os.remove(os.path.join(FOLDER, file))
else:
    os.makedirs(FOLDER)


with open(TEMPLATE_FILE, 'r') as file:
    template_content = file.readlines()


def read_cas5_out(c5_out_filepath):
    output_dict = {}
    
    with open(c5_out_filepath, 'r') as file:
        lines = file.readlines()
    
    # Locate the summary section
    summary_index = None
    for i, line in enumerate(lines):
        if "** C A S M O 5  Summary **" in line:
            summary_index = i
            break
    
    if summary_index is None:
        print("Warning: CASMO5 summary section not found in file.")
        return None
    
    # Find the column names starting with "No"
    column_names = []
    data_start_index = None
    for i in range(summary_index, len(lines)):
        if lines[i].strip().startswith("No"):
            column_names = re.split(r'\s+', lines[i].strip())
            column_names = [col for col in column_names if "Pu" not in col]  # Remove columns containing "Pu"
            data_start_index = i + 1
            break
    
    i = 0
    for col in column_names:
        if col == 'Pin':
            column_names[i] = 'Pin Peak'
        if col == 'Fiss' or col == 'Tot':
            column_names[i] = col + ' Pu'
        i = i + 1
    
    if not column_names or data_start_index is None:
        print("Warning: Column header line not found after CASMO5 summary section.")
        return None
    
    # Process the data
    last_values = {}
    first_entry_found = False
    index = 0
    for i in range(data_start_index, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        parts = re.split(r'\s+', line)
        
        # First entry should be an integer (record index number)
        prev_index = index
        try:
            index = int(parts[0])
            if index<prev_index:
                break

        except ValueError:
            continue
        
        full_data = {}
        
        # First occurrence of No=1 contains all values
        if index == 1 and not first_entry_found:
            for j, col in enumerate(column_names):
                full_data[col] = parts[j] if j < len(parts) else ''
                last_values[col] = full_data[col]  # Store as last valid values
            first_entry_found = True
        else:
            # Fill missing values with previous ones
            for j, col in enumerate(column_names):
                if j < len(parts) and parts[j]:
                    full_data[col] = parts[j]
                    last_values[col] = parts[j]  # Store last valid value
                else:
                    full_data[col] = last_values.get(col, '')  # Use last recorded value if missing
        
        output_dict[index] = full_data
    
    return output_dict

def parse_lattice_data(solution_list):
    lattice_data = {}
    for i, data in enumerate(solution_list):
        key = KEYS[i]
        lattice_data[key] = float(data)
    return lattice_data


def gen_lattice_inp(template_content, folder_name, lattice_name, lattice_data):
    """
    Function to generate a perturbed lattice input file based on pre-generated lattice data.

    Parameters:
        template_content (list): List of strings representing template file lines.
        folder_name (str): Directory where the new lattice file will be saved.
        lattice_name (str): Name of the lattice.
        lattice_data (dict): Dictionary containing pre-generated perturbation data.

    Returns:
        str: Path to the generated lattice input file.
    """
    # Create the output file path
    output_file = os.path.join(folder_name, f"{lattice_name}.BOC.inp")

    # Apply perturbations based on the provided lattice data
    perturbed_content = []
    for line in template_content:
        if line.startswith("FUE"):
            parts = line.split()
            fue_num = int(parts[1])

            # Apply enrichment perturbation if available
            if f"FUE{fue_num}_enr" in lattice_data:
                parts[4] = f"{lattice_data[f'FUE{fue_num}_enr']:.1f}"

            # Apply gads perturbation if available
            if f"FUE{fue_num}_gads" in lattice_data:
                parts[-1] = f"64016={lattice_data[f'FUE{fue_num}_gads']:.1f}"
            
            line = " ".join(parts) + "\n"

        elif line.startswith("IDE"):
            line = f"IDE='{lattice_name}'\n"

        elif line.startswith("SIM"):
            line = f"SIM '{lattice_name}'\n"

        perturbed_content.append(line)
        
    # Write the perturbed content to the new file
    with open(output_file, 'w') as file:
        file.writelines(perturbed_content)
    #print(f"Written perturbed input file: {output_file}")

    return output_file
    

def eval_cas5_boc(file_path):

    folder_name, file_name = os.path.split(file_path)
    log_file = file_name[:-3] + 'log'

    if os.path.exists(folder_name) and os.path.exists(os.path.join(folder_name, file_name)):
        #print(f"Running simulation for {file_path} in background...")
        #print("current directory", os.getcwd())
        os.chdir(folder_name)

        with open(log_file, "w") as log:
            process = subprocess.Popen(["cas5", file_name], stdin=subprocess.PIPE, stdout=log, stderr=log)
            process.stdin.write(b"y\n")
            process.stdin.close()

        time.sleep(5)
        # Wait until no files starting with "tmp" exist
        while any(f.startswith('tmp.'+file_name[:-3]) for f in os.listdir('.')):
            #print(f"Waiting for temporary files to be cleared for {file_path}...")
            time.sleep(1)

        os.chdir("..")
    else:
        print(f"Directory or file {os.path.join(folder_name, file_name)} does not exist.")
    
    outfile_name = file_name[:-3] + 'out'
    c5_out_filepath = os.path.join(folder_name, outfile_name)
    output_dict = read_cas5_out(c5_out_filepath)

    kinf = float(output_dict[1]['K-inf'])
    ppf = float(output_dict[1]['Pin Peak'])

    return kinf, ppf

def calc_score(kinf, ppf, target_kinf=1.05, target_ppf=1.33, kinf_tol=0.001, ppf_tol=0.00):

    kinf_penalty = 0
    ppf_penalty = 0
    if kinf < (target_kinf - kinf_tol):
        kinf_penalty = 2000*(target_kinf-kinf)
    if kinf > (target_kinf + kinf_tol):
        kinf_penalty = 2000*(kinf-target_kinf)
    if ppf > (target_ppf + ppf_tol):
        ppf_penalty = 1000*(ppf-target_ppf)

    score = 100-kinf_penalty-ppf_penalty
    score = round(score, 2)
    return score


def objective_function(solution_list, template_content):
    lattice_data = parse_lattice_data(solution_list)
    lat_index = random.randint(0, 1000)
    file_path = gen_lattice_inp(template_content, FOLDER, lattice_name=f"GE14_{lat_index:03d}", lattice_data=lattice_data)
    kinf, ppf = eval_cas5_boc(file_path)
    score = calc_score(kinf, ppf)

    return score

