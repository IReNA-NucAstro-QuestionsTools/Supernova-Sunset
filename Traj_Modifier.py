input_file = "Traj_CPR2.txt"
output_file = "Traj_CPR2_modified.txt"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if line.strip().endswith("0.41"):
            parts = line.rstrip().split()
            parts[-1] = "0.60"
            outfile.write("    ".join(parts) + "\n")
        else:
            outfile.write(line)
