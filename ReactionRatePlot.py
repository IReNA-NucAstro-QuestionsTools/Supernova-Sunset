import numpy as np
import matplotlib.pyplot as plt

def thielemann_rate(T9, a):
    """Compute reaction rate using Thielemann et al. fit formula.
    
    Parameters:
    - T9: Temperature in GK (can be numpy array)
    - a: List or array of 7 REACLIB coefficients a0 through a6

    Returns:
    - Reaction rate as a numpy array
    """
    assert len(a) == 7, "Expected 7 coefficients for REACLIB fit."
    
    # Compute the sum of a1..a5 * T9**((2i - 5)/3)
    powers = np.array([(2 * i - 5) / 3 for i in range(1, 6)])  # exponents for i=1 to 5
    sum_terms = sum(a[i] * T9 ** powers[i - 1] for i in range(1, 6))
    
    rate = np.exp(a[0] + sum_terms + a[6] * np.log(T9))
    return rate

# Example: User inputs coefficients
a_coeffs = [3.772620e+01, 0.000000e+00, -3.904870e+01, -3.392410e-01, -2.873070e+00 ,3.823690e-01 , -6.666670e-01]  # Replace with your REACLIB values

# Temperature grid (in GK)
T9_vals = np.linspace(-2, 20, 1000)
rates = thielemann_rate(T9_vals, a_coeffs)

base_rate = thielemann_rate(T9_vals, a_coeffs)

# Scale rates up/down by factor of 100
rate_up = base_rate * 100
rate_down = base_rate / 100

# Plot
plt.figure(figsize=(8,6))
plt.plot(T9_vals, rates, label='Reaction Rate')
plt.plot(T9_vals, base_rate, label='Nominal Rate', color='blue')
plt.plot(T9_vals, rate_up, '--', label='Rate × 100', color='red')
plt.plot(T9_vals, rate_down, '--', label='Rate ÷ 100', color='green')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Temperature $T_9$ [GK]')
plt.ylabel('Reaction Rate')
plt.title('Reaction Rate vs Temperature (Thielemann Fit)')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
