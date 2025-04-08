import numpy as np

def f1(x):
    return np.exp(x) * np.sin(4 * x)

a1, b1, h = 1, 2, 0.1
x_vals = np.arange(a1, b1 + h, h)

# Composite Trapezoidal Rule
trapezoidal_result = h * (0.5 * f1(a1) + sum(f1(x_vals[1:-1])) + 0.5 * f1(b1))

# Composite Simpson's Rule
if len(x_vals) % 2 == 0:
    x_vals = x_vals[:-1]  # 確保是奇數點數
simpson_result = h / 3 * (f1(x_vals[0]) + 
                          4 * sum(f1(x_vals[1:-1:2])) + 
                          2 * sum(f1(x_vals[2:-2:2])) + 
                          f1(x_vals[-1]))

# Composite Midpoint Rule
midpoints = (x_vals[:-1] + x_vals[1:]) / 2
midpoint_result = h * sum(f1(midpoints))

print(trapezoidal_result, simpson_result, midpoint_result)
