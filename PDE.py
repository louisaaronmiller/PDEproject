import numpy as np
import matplotlib.pyplot as plt
import os


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

    x_vals = np.linspace(-L/2,L/2 , Nx) #maybe 0,L
    t_vals = np.arange(0, Nt, 0.005)
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

def Analytical2(L, Nx,Nt, M,D=1e5,C=1e8):

    x_vals = np.linspace(-L/2,L/2 , Nx) #maybe 0,L
    t_vals = np.arange(0, Nt, 0.05)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    for n in range(1, M +1):
        eval = (D*(((2*n +1)**2)*(np.pi **2)) - C*L**2)/(L **2)
        u += (
            (2 / L)
            * np.exp(-(((eval) * t)))
            * np.cos(((2*n + 1)* np.pi * x)/(L))
        )

    return x, t, u


def B(alpha,beta,N):
    main_diag = beta * np.ones(N)
    off_diag = alpha * np.ones(N-1)
    B = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return B

def Numerical(L,tmax,deltat,Nx=101, D=1,C=1):
    
    
    xvals = np.linspace(-L/2,L/2,Nx)
    deltax = L/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax + deltat,deltat)
    Nt = len(tvals)
    
    x_in = xvals[1:-1] # defining an interior (not including boundary conditions)
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    initial_u = np.zeros(N_in)
    initial_u[int(len(initial_u)/2)] = 1.0/deltax
    
    alpha = D * deltat / deltax**2
    beta = 1.0 + C * deltat - 2.0 * alpha
    
    B_matrix = B(alpha,beta,N_in) # doesn't include boundary conditions
    u_n = np.array(initial_u) # making code easier to read
    U_in = [u_n.copy()] # doesn't include boundary conditions
    #U.append(u_n.copy())
    for _ in range(Nt-1):
        u_np1 = B_matrix @ u_n
        U_in.append(u_np1.copy())
        u_n = u_np1
    if deltat > ((deltax)**2)/2:
        print('warning deltat not small enough/dx too big')
    U_full = np.zeros((Nt,Nx))
    U_full[:,1:-1] = np.array(U_in) # adding on boundary condition results i.e. 0 at start of x and end x spectrum (-L/2,L/2)
    return x,t,U_full

if False:
    path = r'C:\Users\fowar\OneDrive\Desktop\Folder\university\picgif4'
    j = 1
    for i in np.linspace(2,10,100):    
        #xn,tn,un = Numerical(L=i,tmax = 5,deltat = 0.00005)
        x,t,u = Analytical(L=i,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(x, t, u, cmap='viridis',alpha=0.9)  # edgecolor = 'blue'

        ax.set_title(f'Analytical Diffusion Equation: L = {round(i,3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')

        ax.view_init(elev=20, azim=-60) # angle of the graph

        filename = f"CriticalLength_L{round(j)}.png"
        file_path = os.path.join(path,filename)
        print(f'graph done!')
        j += 1

        plt.savefig(file_path)
        plt.close()

        
'''
xn,tn,un = Numerical(L=10,tmax = 5,deltat = 0.0005)

x,t,u = Analytical(L=10,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

x2,t2,u2 = Analytical2(L=20,Nx = 100,Nt= 5,M= 15, D=1,C=1)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xn, tn, un, cmap='viridis',alpha=0.9)  # edgecolor = 'blue'
ax.plot_surface(x, t, u, cmap='viridis',alpha=0.9)
#ax.plot_surface(x2,t2,u2,cmap='viridis', alpha=0.9)


ax.set_title('Analytical Diffusion Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

ax.view_init(elev=20, azim=-20) # angle of the graph

plt.show()
'''
