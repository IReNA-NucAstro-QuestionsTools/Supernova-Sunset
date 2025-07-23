import numpy as np
import matplotlib.pyplot as plt

# === Load finab.dat ===
# Assumes the file starts with a header or line numbers (we'll skip them)
data = np.loadtxt("finab.dat", comments="#", usecols=(0, 3))  # Columns: A, Xi=4, 3 = Yi
#\\wsl.localhost\Ubuntu\usr\Sean\Programs\WinNet\runs
# Split into arrays
A = data[:, 0].astype(int)
Xi = data[:, 1]

# Sort by mass number for nice plotting
sorted_indices = np.argsort(A)
A = A[sorted_indices]
Xi = Xi[sorted_indices]

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.semilogy(A, Xi, 'o-', markersize=4, label='Mass Fraction $X_i$')

plt.xlabel("Mass Number $A$")
plt.ylabel("Mass Fraction $X_i$")
plt.title("Final Mass Fractions from Winnet Output")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
