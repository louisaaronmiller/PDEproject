import numpy as np
import matplotlib.pyplot as plt
import os

#region ------------------------------------------------------------------------ QUESTION ONE ------------------------------------------------------------------------

# ------------------------- Functions -------------------------
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

def B_partbFAIL(alpha,beta,N):
    main_diag = beta * np.ones(N)
    off_diag = 2*alpha * np.ones(N-1)
    B = np.diag(main_diag) + np.diag(off_diag,k=1)
    return B

def B_partb(alpha,beta,N):
    B = np.zeros((N, N), dtype=float)
    np.fill_diagonal(B, beta)
    
    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha
    B[0, 1] = 2.0 * alpha
    B[0, 1] = 2.0 * alpha
    return B

def indexfinder(xvals,L):
    '''
    This function returns the index of the array where the first value is
    greater than -L/2 and the returns the last index corresponding to the
    x value that is less than L/2
    
    This is mainly only for x values with ranges expanded by a constant alpha say,
    part D of project.
    '''
    firstindex = None
    secondindex = None
    
    for key,x in enumerate(xvals):
        if firstindex is None and x >= -L/2: # first x value that is inside the range
            firstindex = key
        if x <= L/2: # keys updating until the range x is out of the range, therefore the last x value's index
            secondindex = key
    return (firstindex, secondindex)

def B_partd(N,firstindex,secondindex,D,C,deltat,deltax):
    
    alpha = D * deltat / deltax**2
    beta = 1.0 + C * deltat - 2.0 * alpha
    
    beta0 = 1.0 - 2.0 * alpha # used when |x| > L/2
    
    main_diag = beta0 * np.ones(N) # creates an array with betas the same size as the diaganol in the matrix
    main_diag[firstindex:secondindex +1] = beta # changes the array within this range to beta0 eg. [beta,beta,beta,beta0,beta0,beta,beta,beta]
    
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

def Numerical_partb(L,tmax,deltat,Nx=101, D=1,C=1):
    
    xvals = np.linspace(-L/2,L/2,Nx)
    deltax = L/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax + deltat,deltat)
    Nt = len(tvals)
    
    x_in = xvals[:-1] # This INCLUDES the first column, however, its not filled with zeros. The last column is.
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    initial_u = np.zeros(N_in)
    initial_u[int(len(initial_u)/2)] = 1.0/deltax
    
    alpha = D * deltat / deltax**2
    beta = 1.0 + C * deltat - 2.0 * alpha
    
    B_matrix = B_partb(alpha,beta,N_in) # doesn't include L/2 boundary condition (includes -L/2 Neumann)
    u_n = np.array(initial_u) # making code easier to read
    U_in = [u_n.copy()]
    
    for _ in range(Nt- 1): #Nt -1
        u_np1 = B_matrix @ u_n
        U_in.append(u_np1.copy())
        u_n = u_np1
        
    if deltat > ((deltax)**2)/2:
        print('warning deltat not small enough/dx too big')
    U_full = np.zeros((Nt,Nx))
    U_full[:,:-1] = np.array(U_in) # adding on boundary condition results (zero column to end)
    # It doesn't really 'add' I have created a background of zeros and we are just slotting in leaving the last column of zeros there.
    return x,t,U_full

def AnalyticalC(L, Nx,Nt, M,D=1e5,C=1e8):
    x_vals = np.linspace(-L/2,L/2 , Nx) #maybe 0,L
    t_vals = np.arange(0, Nt, 0.005)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    for n in range(0, M +1):
        eval = (D*((n*np.pi)/L)**2) - C
        u += (
            (2 / L)
            * np.cos((np.pi * n) / 2)
            * np.exp(-(((eval) * t)))
            * np.cos((n * np.pi * (x + L / 2)) / (L))
        )

    return x, t, u

def NumericalD(L,tmax,deltat,a,Nx=101, D=1,C=1):
    '''
    a = a constant that multiplies the range if L 
    '''

    xvals = np.linspace(-a * (L/2), a * (L/2),Nx)
    deltax = (a*L)/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax + deltat,deltat) # + deltat to make sure everything syncs
    Nt = len(tvals)
    
    x_in = xvals[1:-1] # defining an interior (not including boundary conditions)
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    initial_u = np.zeros(N_in)
    initial_u[int(len(initial_u)/2)] = 1.0/deltax
    
    firstindex, secondindex = indexfinder(xvals,L)
    firstindex -= 1 # This is because i included the whole range of xvals above, but im actually only using x_in, but can't slice, since we need to keep the index based on xvals
    secondindex -= 1
    B_matrix = B_partd(N_in,firstindex,secondindex,D,C,deltat,deltax) # doesn't include boundary conditions
    
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


#region ------------------------- Results -------------------------

xb,tb,ub = Numerical_partb(L=3,tmax=5,deltat = 0.00005)

xn,tn,un = Numerical(L=10,tmax = 5,deltat = 0.0005)

x,t,u = Analytical(L=10,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

x2,t2,u2 = Analytical2(L=20,Nx = 100,Nt= 5,M= 15, D=1,C=1)

xd,td,ud = NumericalD(L=1.12,tmax = 5, deltat = 0.00005,a=2)

xC,tC,uC = AnalyticalC(L=1,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

#endregion

#region ------------------------- Animations -------------------------

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

if False:
    path = r"C:\Users\40298094\OneDrive - Queen's University Belfast\Project\1partbgraph"
    j = 1
    for i in np.linspace(3,150,200):
        
        xb,tb,ub = Numerical_partb(L=i,tmax=5,deltat = 0.0005)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xb,tb,ub,cmap='viridis', alpha=0.9)
    
    
        ax.set_title(f'Numerical Diffusion Equation L={round(i,3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
    
        ax.view_init(elev=20, azim=-50) # angle of the graph
    
        filename = f"partbCriticalL{round(j)}.png"
        file_path = os.path.join(path,filename)
        print(f'graph done! {j}')
        j += 1

        plt.savefig(file_path)
        plt.close()

if False: # This shows that partb is the same L approximately equal to pi when it starts.
    for i in np.linspace(2.5,5,50):
         xb,tb,ub = Numerical_partb(L=i,tmax=5,deltat = 0.0005)
         
         fig = plt.figure()
         ax = plt.axes(projection='3d')
         ax.plot_surface(xb,tb,ub,cmap='viridis', alpha=0.9)
     
     
         ax.set_title(f'Numerical Diffusion Equation L={round(i,3)}')
         ax.set_xlabel('x')
         ax.set_ylabel('t')
         ax.set_zlabel('u(x,t)')
     
         ax.view_init(elev=20, azim=-50) # angle of the graph
    
         plt.show()
         
if False:
    for i in np.linspace(2.5,5,50):
         xd,td,ud = NumericalD(L=i,tmax=5,deltat = 0.00005,a=6)
         
         fig = plt.figure()
         ax = plt.axes(projection='3d')
         ax.plot_surface(xd,td,ud,cmap='viridis', alpha=0.9)
     
     
         ax.set_title(f'Numerical Diffusion Equation L={round(i,3)}')
         ax.set_xlabel('x')
         ax.set_ylabel('t')
         ax.set_zlabel('u(x,t)')
     
         ax.view_init(elev=20, azim=-50) # angle of the graph
    
         plt.show()

#endregion

# ------------------------- Normal Plotting -------------------------



fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.plot_surface(xn, tn, un, cmap='viridis',alpha=0.9)  # edgecolor = 'blue'
#ax.plot_surface(x, t, u, cmap='viridis',alpha=0.9)
#ax.plot_surface(x2,t2,u2,cmap='viridis', alpha=0.9)
#ax.plot_surface(xb,tb,ub,cmap='viridis', alpha=0.9)
#ax.plot_surface(xC,tC,uC,cmap='viridis', alpha=0.9)
ax.plot_surface(xd,td,ud,cmap='viridis', alpha=0.9)


#ax.set_title('Analytical Diffusion Equation')
ax.set_title('Numerical Diffusion Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

ax.view_init(elev=20, azim=-20) # angle of the graph

plt.close()

#endregion

#region ------------------------------------------------------------------------ QUESTION TWO ------------------------------------------------------------------------

#region ------------------------- Functions -------------------------

def AnalyticalWave(Nx,Nt, M=15):
    '''
    Wave equation, n can't be equal to 16 in the summation due to Dn.
    Dn has a term with a denominator n-16, thus when n=16 this is infinite or
    undefined, thus approximations up to 16 will have to do.

    '''

    x_vals = np.linspace(0,2 *np.pi , Nx)
    t_vals = np.arange(0, Nt, 0.005)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    for n in range(1, M +1):
        if n == 16:
            Bn = ((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2)
            Dn = 0
            u += (Bn * np.cos(n*t) + Dn * np.sin(n*t)) * np.sin((n * x)/2)
            continue
        Bn = ((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2)
        Dn = (1/(n * np.pi)) * (((2 * (np.pi ** 2)* (-1)**n) * (1/(n+16) + 1/(n-16) -2/n)) + (((-1)**n) -1)* (8/(n**3) -4/((n+16)**3)-4/((n-16)**3)))
        u += (Bn * np.cos(n*t) + Dn * np.sin(n*t)) * np.sin((n * x)/2)

    return x, t, u

#endregion
#region ------------------------- Results -------------------------

xaw,taw,uaw = AnalyticalWave(100,5)

#endregion
#region ------------------------- Plotting -------------------------

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(xaw,taw,uaw,cmap='viridis', alpha=0.9)

ax.set_title('Analytical Wave Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

ax.view_init(elev=20, azim=-10) # angle of the graph

plt.show()

#endregion

































#endregion
