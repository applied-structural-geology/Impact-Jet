import numpy as np

def quad_teos(a, b, d, P, m, A, B, Eo):
    c = (Eo*(m+1)**2)
    d = (m+1)*d
    quad_a = (a*d)/c
    quad_b = (a*d + b*d + ((A*m)/c) + (B*(m)**2)/c - P/c)
    quad_c = A*m + B*(m)**2 - P
    E = np.round((-1 * quad_b + np.sqrt((quad_b ** 2 - 4 * quad_a * quad_c))) / (2 * quad_a),4)
    return E

def f(x, y, a, b, m, A, B, Eo):
    return (1 / x ** 2) * ((a + (b * Eo * (m + 1) ** 2) / (y + Eo * (m + 1) ** 2)) * x * y + A * m + B * m ** 2)

def rk4_step(f, x, y, h, a, b, m, A, B, Eo):
    k1 = h * f(x, y, a, b, m, A, B, Eo)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1, a, b, m, A, B, Eo)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2, a, b, m, A, B, Eo)
    k4 = h * f(x + h, y + k3, a, b, m, A, B, Eo)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def solve_rk4(f, x0, y0, h, x_end,  a, b, m, A, B, Eo):
    while x0 < x_end:
        h = min(h, x_end - x0)
        y0 = rk4_step(f, x0, y0, h, a, b, m, A, B, Eo)
        x0 += h
    return y0

def solve_tillotson(a, b, d, Pf, m, A, B, Eo, C):
    E = quad_teos(a, b, d, Pf, m, A, B, Eo)
    x0, y0 = d, 0  # Initial condition
    h = 0.0001  # Step size
    x_end = (m+1)*d  # End point
    E_c = solve_rk4(f, x0, y0, h, x_end,  a, b, m, A, B, Eo)
    temp = (E +E_c)/C
    return temp