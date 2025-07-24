def extract_first_last_ye(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove comment lines and empty lines
    data_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]

    if not data_lines:
        raise ValueError("No data lines found in file.")

    # Extract Ye values (5th column, index 4)
    first_ye = float(data_lines[0].split()[4])
    last_ye = float(data_lines[-1].split()[4])

    return first_ye, last_ye

# Example usage
if __name__ == "__main__":
    filepath = "mainout.dat"
    first_ye, last_ye = extract_first_last_ye(filepath)
    print(f"First Ye: {first_ye:.4f}")
    print(f"Last Ye:  {last_ye:.4f}")
