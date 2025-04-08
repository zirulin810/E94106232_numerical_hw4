import numpy as np
def f4a_transformed(t):
    return 4 * t**2 * np.sin(t**4)

def composite_simpson(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h / 3 * (f(x[0]) + 
                    4 * np.sum(f(x[1:-1:2])) + 
                    2 * np.sum(f(x[2:-2:2])) + 
                    f(x[-1]))

simpson_4a = composite_simpson(f4a_transformed, 0, 1, 4)

def f4b_transformed(t):
    return t**6 * np.sin(1 / t)

epsilon = 1e-6
simpson_4b = composite_simpson(f4b_transformed, epsilon, 1, 4)

print("4a: ", simpson_4a)
print("4b: ", simpson_4b)
