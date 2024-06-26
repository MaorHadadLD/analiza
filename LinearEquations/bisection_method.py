import numpy as np

def max_steps(a, b, err):
    s = int(np.floor(-np.log2(err / (b - a)) / np.log2(2) - 1))
    return s

def bisection_method(f, a, b, tol=1e-6):
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")
    c, k = 0, 0
    steps = max_steps(a, b, tol)

    print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    while abs(b - a) > tol and k < steps:
        c = a + (b - a) / 2

        if f(c) == 0:
            return c

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

        print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
        k += 1

    return c

if __name__ == '__main__':
    f = lambda x: (6*x**3 + x**2 + 2) / (2*x - 6)
    roots1 = bisection_method(f, -3, 0)
    # roots2 = bisection_method(f, 0, -3)
    print(f"\nThe equation f(x) has approximate roots at x = {roots1}")
