import numpy as np
import matplotlib.pyplot as plt

def B(alpha,beta,N):
    main_diag = beta * np.ones(N)
    off_diag = alpha * np.ones(N-1)
    B = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    return B

def Numerical(L,tmax,deltat,Nx=101, D=1,C=1):
    
    xvals = np.linspace(-L/2,L/2,Nx)
    deltax = L/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax + deltat,deltat) # + deltat to make sure everything syncs
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

    for _ in range(Nt-1):
        u_np1 = B_matrix @ u_n # @ is matrix mulitplication 
        U_in.append(u_np1.copy())
        u_n = u_np1
    if deltat > ((deltax)**2)/2:
        print('warning deltat not small enough/dx too big')
    U_full = np.zeros((Nt,Nx))
    U_full[:,1:-1] = np.array(U_in) # putting the interior back in i.e. "adding" columns for 0 at start of x and end x spectrum (-L/2,L/2)
    return x,t,U_full

xn,tn,un = Numerical(L=10,tmax = 5,deltat = 0.0005)

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xn, tn, un, cmap='viridis',alpha=0.9)
ax.set_title('Numerical Diffusion Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

ax.view_init(elev=20, azim=-20) # angle of the graph

plt.show()
