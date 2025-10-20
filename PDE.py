import numpy as np


def Analytical(L, N):

    x_vals = np.linspace(-L / 2, L / 2, N)
    t_vals = np.arange(0, N, 1)
    u_vals = []
    vals = dict(zip(t_vals, x_vals))
    for x, t in vals.items():
        running_total = 0
        for n in range(1, N + 1):
            u = (
                (2 / L)
                * np.sin((np.pi * n) / 2)
                * np.exp(-(n**2 * np.pi**2 - L**2) * t)
                * np.sin((n * np.pi * (x + L / 2)) / (L))
            )
            running_total += u
        u_vals.append(running_total)

    return x_vals, t_vals, u_vals
