import numpy as np
import matplotlib.pyplot as plt


def Analytical(L, Nx,Nt, M,D=1e5,C=1e8):
    '''

    u(x=L/2,t) = u(x=-L/2,t) = 0

    An aditional for loop isn't needed within the for loop over (1, M+1)
    for x and t values since numpy arrays are used,
    and they are the same shape as eachother.
    Numpy is broadcasts each value 

    x vals here are a matrix where each row goes from -L/2 to L/2
    (every row is idential):

    [-L/2,...,L/2],[-L/2,...,L/2],...,[-L/2,...,L/2]

    t vals are also a matrix where each row
    is a single value repeated i.e.

    [0,0,...,0],[Δt,Δt,...,Δt],...,[N_tΔt,N_tΔt,...,N_tΔt]

    using numpy broadcasting u values are represented as:

    [u(x=-L/2,t=0),u(x=-L/2 + Δx,t=0),...,u(x=L/2,t=0)],
    [u(x=-L/2,t=Δt),u(x=-L/2 + Δx,t=Δt),...,u(x=L/2,t=Δt)],
    .
    .
    .
    [u(x=-L/2,t=N_tΔt),u(x=-L/2 + Δx,t=N_tΔt),...,u(x=L/2,t=N_tΔt)]

    '''

    x_vals = np.linspace(-L / 2, L / 2, Nx)
    t_vals = np.arange(0, Nt, 0.05)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    for n in range(1, M +1):
        eval = (D*((n*np.pi)/L)**2) - C
        u += (
            (2 / L)
            * np.sin((np.pi * n) / 2)
            * np.exp(-(((eval) * t)))
            * np.sin((n * np.pi * (x + L / 2)) / (L))
        )

    return x, t, u

def B(alpha,beta,N):
    main_diag = beta * np.ones(n)
    off_diag = alpha * np.ones(n-1)
    B = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return B

def Numerical(L,tmax,deltat,Nx=101):
    xvals = np.linspace(-L/2,L/2,Nx)
    xvals_e = dict(enumerate(xvals))
    deltax = L/(2*(Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax,deltat)
    x,t = np.meshgrid(xvals,tvals)
    print(xvals_e)
    return xvals


Numerical(10,5,0.05)
    
    





x,t,u = Analytical(L=10,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x,t,u,cmap='viridis') #edgecolor = 'blue'
ax.set_title('Analytical Diffusion Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()
