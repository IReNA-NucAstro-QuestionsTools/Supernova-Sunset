import numpy as np

Cross_sec_Ratio=8.35082e49
Cons_L=3e51/Cross_sec_Ratio
Cons_E=25.88*Cross_sec_Ratio


input_file = "Traj_CPR2.txt"
ye_values = np.round(np.arange(0.20, 0.95, 0.05), 2)

# Read the original file once
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# Loop through each Ye value
for ye in ye_values:
    ye_str = f"{ye:.2f}"
    output_file = f"Traj_CPR2_Ye_{ye_str}.txt"

    with open(output_file, 'w') as outfile:
        for line in lines:
            # Update the header line if it contains "Ye ="
            if "Ye =" in line:
                parts = line.split("Ye =")
                new_line = f"{parts[0]}Ye = {ye_str} at T = 10 GK\n"
                outfile.write(new_line)
            # Replace data lines ending in a numeric value
            elif line.strip() and line.strip().split()[-1].replace('.', '', 1).isdigit():
                parts = line.rstrip().split()
                parts[-1] = ye_str
                outfile.write("    ".join(parts) + "\n")
            else:
                outfile.write(line)

    print(f"Created {output_file}")
