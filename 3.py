import numpy as np
import numpy.polynomial.legendre as leg
from scipy.integrate import quad

def f3(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

# a. Composite Simpson's Rule（4x4）
def composite_simpson_double(f, ax, bx, ay_func, by_func, nx, ny):
    hx = (bx - ax) / nx
    result = 0
    for i in range(0, nx + 1):
        x = ax + i * hx
        wx = 4 if i % 2 == 1 else 2
        if i == 0 or i == nx:
            wx = 1
        ay, by = ay_func(x), by_func(x)
        hy = (by - ay) / ny
        inner = 0
        for j in range(0, ny + 1):
            y = ay + j * hy
            wy = 4 if j % 2 == 1 else 2
            if j == 0 or j == ny:
                wy = 1
            inner += wy * f(x, y)
        result += wx * inner * hy / 3
    return result * hx / 3

a3, b3 = 0, np.pi / 4
ay_func = lambda x: np.sin(x)
by_func = lambda x: np.cos(x)
simpson_double_result = composite_simpson_double(f3, a3, b3, ay_func, by_func, 4, 4)

# b. Gaussian Quadrature (n=3, m=3)
def gaussian_quadrature_double(f, ax, bx, ay_func, by_func, nx, ny):
    [xi, wi] = leg.leggauss(nx)
    [yi, wj] = leg.leggauss(ny)
    result = 0
    for i in range(nx):
        x = 0.5 * (bx - ax) * xi[i] + 0.5 * (bx + ax)
        ay, by = ay_func(x), by_func(x)
        for j in range(ny):
            y = 0.5 * (by - ay) * yi[j] + 0.5 * (by + ay)
            result += wi[i] * wj[j] * f(x, y) * (by - ay) / 2
    return result * (bx - ax) / 2

gauss_double_result = gaussian_quadrature_double(f3, a3, b3, ay_func, by_func, 3, 3)

def integrand(y, x):
    return f3(x, y)

true_double_value, _ = quad(lambda x: quad(lambda y: integrand(y, x), 0, np.cos(2 * x))[0], 0, np.pi / 4)

error_n3 = abs(simpson_double_result - true_double_value)
error_n4 = abs(gauss_double_result - true_double_value)

print("Composite Simpson's Rule: ", simpson_double_result)
print("Gaussian Quadrature: ", gauss_double_result)
print("Error of Composite Simpson's Rule: ", error_n3)
print("Error of Gaussian Quadrature: ", error_n4)
