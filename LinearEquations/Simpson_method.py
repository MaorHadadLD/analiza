import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify

x = sp.symbols('x')

def simpsons_rule(f, a, b, n):
    """
    Simpson's Rule for Numerical Integration

    Parameters:
    f (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals (must be even).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's Rule.")

    h = (b - a) / n

    integral = f(a) + f(b)  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x_i)
        else:
            integral += 4 * f(x_i)

    integral *= h / 3

    return integral

"""
Date:08/04/2024
Groups: Raphael Benoliel 209946854
Daniel Vaknin 314753161
Maor Hadad 312469463
Bar Cohen 316164938
name: Maor Hadad 312469463
"""
if __name__ == '__main__':
    f = lambda x: (2*x + math.cos(x**3 + 2*x**2 - 6)) / (x + 2*math.e**-2*x)
    n = 34
    a = -2.9
    b = -1.4

    print(f"Division into n={n} sections ")
    integral = simpsons_rule(f, a, b, n)
    print(f"Numerical Integration of definite integral in range [{a},{b}] is {integral:.5f}")
