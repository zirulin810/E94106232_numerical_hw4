import numpy as np
import numpy.polynomial.legendre as leg
from scipy.integrate import quad

def f2(x):
    return np.log(x) * x

a2, b2 = 1, 1.5

def gaussian_quadrature(f, a, b, n):
    [x, w] = leg.leggauss(n)
    t = 0.5 * (x + 1) * (b - a) + a
    return 0.5 * (b - a) * np.sum(w * f(t))

gauss_n3 = gaussian_quadrature(f2, a2, b2, 3)
gauss_n4 = gaussian_quadrature(f2, a2, b2, 4)

true_value, _ = quad(f2, a2, b2)

error_n3 = abs(gauss_n3 - true_value)
error_n4 = abs(gauss_n4 - true_value)

print("n=3: ", gauss_n3, "error: ", error_n3)
print("n=4: ", gauss_n4, "error: ", error_n4)
