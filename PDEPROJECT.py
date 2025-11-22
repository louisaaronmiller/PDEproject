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

    for n in range(0, M +1):
        k_m = ((2 * n + 1) * np.pi)/(2 * L)
        eval = C-(D * k_m**2)
        u += (
            (2 / L)
            * np.cos((k_m * L)/2)
            * np.exp(((eval) * t))
            * np.cos(k_m * (x + L/2))
        )

    return x, t, u

def findaverage(Lvals,tmax,deltat,Nx=101,D=1,C=1):
    result = []
    for L in Lvals:
        _,_,u = Numerical(L,tmax,deltat,Nx=Nx,D=D,C=C) 
        height = np.mean(u[20000:], axis=1) # [20000:] ignores the 1/deltax spike
        result.append(height)
        print(f'{L} done!')
    return np.mean(result)

def CriticalLengthFinder(Lvals,tmax,deltat,Nx=101,D=1,C=1,threshhold = 1e-6,flag=True):
    '''
    Finds the critical length for Neutron Diffusion PDE
    '''
    if flag:
        for L in Lvals:
            _,_,u = Numerical(L,tmax,deltat,Nx=Nx,D=D,C=C) # change to check critical length for different systems
            if np.mean(np.mean(u[20000:], axis=1)) > threshhold: # [20000:] ignores the 1/deltax spike
                return L
            else:
                print(f'no {L}')
                continue
    else:
        for L in Lvals:
            _,_,u = Numerical(L,tmax,deltat,Nx=Nx,D=D,C=C) # change to check critical length for different systems
            if np.any(np.max(u[20000:], axis=1) > threshhold): # [20000:] ignores the 1/deltax spike
                return L
            else:
                print(f'no {L}')
                continue

def growth_rate(u, skip=20000):
    """
    Compute growth rate lambda by fitting log amplitude vs time.
    """
    A = np.sqrt(np.mean(u[skip:]**2, axis=1))  
    A = np.maximum(A, 1e-20)
    t = np.arange(len(A))
    
    coeff = np.polyfit(t, np.log(A), 1) 
    lambda_est = coeff[0]
    return lambda_est

def CriticalLengthFinder2(Lvals, tmax, deltat, Nx=101, D=1, C=1):
    results = []
    L_arr = []

    for L in Lvals:
        _, _, u = Numerical_partb(L, tmax, deltat, Nx=Nx, D=D, C=C)
        lam = growth_rate(u)

        results.append((lam))
        L_arr.append(L)
        print(f"L={L:.3f}, lambda={lam:.4e}")


    return L_arr,results

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

def B_partb(alpha,beta,N): # uses ghost point - first row is beta, 2alpha,0,...
    B = np.zeros((N, N), dtype=float)
    np.fill_diagonal(B, beta)
    
    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha
    B[0, 1] = 2.0 * alpha
    B[0, 1] = 2.0 * alpha
    return B

def B_partb2ANOTHERFAILQUESTIONMARK(alpha,beta,N):
    B = np.zeros((N, N), dtype=float)

    np.fill_diagonal(B, beta)

    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha

    B[0, 0] = beta + alpha

    return B

def B_partb3(alpha,beta,N): # first row is (beta + alpha), 0, alpha
    if N < 2:
        raise ValueError("N must be >= 2")
    B = np.zeros((N, N), dtype=float)

    # Fill main diagonal
    np.fill_diagonal(B, beta)

    # Add tridiagonals for i>=1
    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha

    # Modify first row / first diagonal element per your request
    B[0, 0] = beta + alpha

    # Set the (0,1) entry to zero (we want 0 there)
    if N > 1:
        B[0, 1] = 0.0

    # Put alpha in (0,2) if that column exists
    if N > 2:
        B[0, 2] = alpha

    return B

def B_partb4(alpha,beta,N): # first row is 0, (beta + alpha), alpha
    if N < 3:
        raise ValueError("N must be >= 3 to match requested pattern.")

    B = np.zeros((N, N), dtype=float)

    # Fill main diagonal
    np.fill_diagonal(B, beta)

    # Fill tridiagonals
    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha

    # Overwrite the first row:
    B[0, :] = 0.0
    B[0, 1] = beta + alpha   # column 1
    B[0, 2] = alpha          # column 2

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
    
    B_matrix = B_partb4(alpha,beta,N_in) # doesn't include L/2 boundary condition (includes -L/2 Neumann)
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

def B_partc(alpha,beta,N):
    if N < 3:
        raise ValueError("N must be >= 3 for this structure.")

    B = np.zeros((N, N), dtype=float)

    # Fill main diagonal
    np.fill_diagonal(B, beta)

    # Fill tridiagonals
    for i in range(1, N):
        B[i, i-1] = alpha
        B[i-1, i] = alpha

    # -------- FIRST ROW --------
    B[0, :] = 0.0
    B[0, 0] = beta + alpha
    B[0, 2] = alpha   # if N > 2, guaranteed since N>=3

    # -------- LAST ROW --------
    B[-1, :] = 0.0
    B[-1, -1] = beta + alpha
    B[-1, -3] = alpha   # the "alpha" before the 0, then (beta+alpha)

    return B

def Numerical_partc(L,tmax,deltat,Nx=101, D=1,C=1):
    
    xvals = np.linspace(-L/2,L/2,Nx)
    deltax = L/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax + deltat,deltat)
    Nt = len(tvals)
    
    x_in = xvals # This all since here we don't apend a column of zeros anywhere.
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    initial_u = np.zeros(N_in)
    initial_u[int(len(initial_u)/2)] = 1.0/deltax
    
    alpha = D * deltat / deltax**2
    beta = 1.0 + C * deltat - 2.0 * alpha
    
    B_matrix = B_partc(alpha,beta,N_in) # doesn't include L/2 boundary condition (includes -L/2 Neumann)
    u_n = np.array(initial_u) # making code easier to read
    U_in = [u_n.copy()]
    
    for _ in range(Nt- 1): #Nt -1
        u_np1 = B_matrix @ u_n
        U_in.append(u_np1.copy())
        u_n = u_np1
        
    if deltat > ((deltax)**2)/2:
        print('warning deltat not small enough/dx too big')
    U_full = np.zeros((Nt,Nx))
    U_full = np.array(U_in) 
    return x,t,U_full

def AnalyticalC(L, Nx,Nt, M,D=1e5,C=1e8):
    x_vals = np.linspace(-L/2,L/2 , Nx) #maybe 0,L
    t_vals = np.arange(0, Nt, 0.005)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    u += (1/L) * np.exp(C * t) # first term.

    for n in range(1, M +1): # from n=1 onwards
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

xn,tn,un = Numerical(L=3.14,tmax = 5,deltat = 0.00005)

x,t,u = Analytical(L=10,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

x2,t2,u2 = Analytical2(L=3,Nx = 100,Nt= 5,M= 15, D=1,C=1)

xd,td,ud = NumericalD(L=1.12,tmax = 5, deltat = 0.00005,a=2)

x3,t3,u3 = AnalyticalC(L=3.14,Nx = 100,Nt = 5,M = 15,D = 1,C = 1)

xC,tC,uC = Numerical_partc(L=3.14,tmax = 5,deltat = 0.00005)

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

if False:
    path = r'C:\Users\fowar\OneDrive\Desktop\Folder\university\picgif5'
    j = 1
    for i in np.linspace(1,3,50):
        xb,tb,ub = Numerical_partb(L=i,tmax=5,deltat = 0.00005)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(xb,tb,ub,cmap='viridis', alpha=0.9)
    
    
        ax.set_title(f'Numerical Diffusion Equation L={round(i,3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
    
        ax.view_init(elev=20, azim=-20) # angle of the graph


        filename = f"CriticalLength_L{round(j)}.png"
        file_path = os.path.join(path,filename)
        print(f'graph done!')
        j += 1

        plt.savefig(file_path)
        plt.close()


if False:
    path = r'C:\Users\fowar\OneDrive\Desktop\Folder\university\picgif6'
    j = 1
    for i in np.linspace(1,3,50):
        x2,t2,u2 = Analytical2(L=i,Nx = 100,Nt= 5,M= 15, D=1,C=1)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(x2,t2,u2,cmap='viridis', alpha=0.9)
    
    
        ax.set_title(f'Analytical Diffusion Equation L={round(i,3)}')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
    
        ax.view_init(elev=20, azim=-20) # angle of the graph


        filename = f"CriticalLength_L{round(j)}.png"
        file_path = os.path.join(path,filename)
        print(f'graph done!')
        j += 1

        plt.savefig(file_path)
        plt.close()

#endregion

# ------------------------- Normal Plotting -------------------------

#L, approx = CriticalLengthFinder2(Lvals=np.linspace(1.54,1.60,100), tmax=5, deltat=0.00005, Nx=101, D=1, C=1)
'''
plt.plot(L,approx)
plt.minorticks_on()
plt.grid(True)
plt.xlabel('$L$')
plt.ylabel('$\\lambda_A$')
plt.title('Numerical Critical Length (Neumann & Dirichlet Boundary Condition)')
plt.xlim(1.55,1.59)
plt.ylim(-0.25e-5,0.25e-5)
plt.axhline(y=0, color='r', linestyle='--')
plt.plot([1.5785,1.5785],[-5, 0],'--k')
plt.scatter([1.5785],[0],color = 'k')
#plt.plot(np.linspace(0,20,len(L)),[0] * len(L), color='r', linestyle='--')
'''
fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.plot_surface(xn, tn, un, cmap='viridis',alpha=0.9)  # edgecolor = 'blue'
#ax.plot_surface(x, t, u, cmap='viridis',alpha=0.9)
#ax.plot_surface(x2,t2,u2,cmap='viridis', alpha=0.9)
#ax.plot_surface(xb,tb,ub,cmap='viridis', alpha=0.9)
ax.plot_surface(xC,tC,uC,cmap='viridis', alpha=0.9)
#ax.plot_surface(x3,t3,u3,cmap='viridis', alpha=0.9)
#ax.plot_surface(xd,td,ud,cmap='viridis', alpha=0.9)


#ax.set_title('Analytical Diffusion Equation')
#ax.set_zlim(0, 17.5)
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
    t_vals = np.arange(0, Nt, 0.0005)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)

    B = np.zeros(M+1)
    D = np.zeros(M+1)


    for n in range(1, M +1):

        Bn = ((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2)
        B[n] = Bn

        if n == 16:
            Dn = (-(3 * np.pi )/256)
            D[n] = Dn
        else:
            Dn = (1/(n * np.pi)) * (((2 * (np.pi ** 2)* (-1)**n) * (1/(n+16) + 1/(n-16) -2/n)) + (((-1)**n) -1)* (8/(n**3) -4/((n+16)**3)-4/((n-16)**3)))
            D[n] = Dn

        u += (Bn * np.cos(n*t) + Dn * np.sin(n*t)) * np.sin((n * x)/2)

    # calculating energy integral stuff

    u_x = np.gradient(u, x_vals, axis=1) # x derivative
    u_t = np.gradient(u, t_vals, axis=0) # t derivative

    integrand = 0.5*(u_t**2 + 4*u_x**2) # energy integral
    Et = np.trapezoid(integrand, x_vals, axis=1)
    return x, t, u, Et

def WavePerturb(Nx,Nt, pert, M=15):
    x_vals = np.linspace(0,2 *np.pi , Nx)
    t_vals = np.arange(0, Nt, 0.005)
    x,t = np.meshgrid(x_vals,t_vals)  
    u = np.zeros_like(x)


    for n in range(1, M +1):

        Bn = (1.0+ pert)* (((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2))

        if n == 16:
            Dn = (-(3 * np.pi )/256)
        else:
            Dn = (1/(n * np.pi)) * (((2 * (np.pi ** 2)* (-1)**n) * (1/(n+16) + 1/(n-16) -2/n)) + (((-1)**n) -1)* (8/(n**3) -4/((n+16)**3)-4/((n-16)**3)))

        u += (Bn * np.cos(n*t) + Dn * np.sin(n*t)) * np.sin((n * x)/2)
    return x, t, u

def indexfinderWave(xvals):
    '''
    This function returns the index of the array where the first value is
    greater than pi - 1 and the returns the last index corresponding to the
    x value that is less than pi + 1
    '''
    firstindex = None
    secondindex = None
    
    for key,x in enumerate(xvals):
        if firstindex is None and x >= np.pi - 1: # first x value that is inside the range
            firstindex = key
        if x <= np.pi + 1: # keys updating until the range x is out of the range, therefore the last x value's index
            secondindex = key
    return (firstindex, secondindex)

def NumericalWave(tmax,deltat,Nx=101):
    
    xvals = np.linspace(0,2 * np.pi,Nx)
    deltax = (2*np.pi)/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax,deltat) # + deltat to make sure everything syncs - normally there but removed for the time being
    Nt = len(tvals)
    
    x_in = xvals[1:-1] # defining an interior (not including boundary conditions)
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    firstindex, secondindex = indexfinderWave(x_in)
    initial_u_0 = np.zeros(N_in)
    for i in range(0,N_in):
        if i >= firstindex and i <= secondindex:
            initial_u_0[i] = 1

    initial_u_1 = np.zeros(N_in)
    for i in range(0,N_in):
        if i >= firstindex and i <= secondindex:
            initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) +1
        else:
            initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2)
    
    alpha = (4 * (deltat)**2) / deltax**2
    beta = 2 * (1 - alpha) 
    
    B_matrix = B(alpha,beta,N_in) # doesn't include boundary conditions
    u_n0 = np.array(initial_u_0) # making code easier to read
    u_n1 = np.array(initial_u_1)
    U_in = [u_n0.copy(), u_n1.copy()] # doesn't include boundary conditions
    #U.append(u_n.copy())
    for _ in range(Nt-2):
        u_np1 = (B_matrix @ u_n1) - u_n0
        U_in.append(u_np1.copy())
        u_n0, u_n1 = u_n1, u_np1
    if deltat > deltax/2:
        print('Δt too large, may be unstable')
    U_full = np.zeros((Nt,Nx))
    U_full[:,1:-1] = np.array(U_in) # adding on boundary condition results i.e. 0 at start of x and end x spectrum (0,2pi)
    return x,t,U_full,deltax

def Bn(N):
    Bn_arr = []
    for n in range(1,N+1):
        Bn_arr.append(((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2))
    return Bn_arr

def Dn(N):
    Dn_arr = []
    for n in range(1,N+1):
        if n==16:
            Dn_arr.append(np.nan)
            continue
        Dn_arr.append((1/(n * np.pi)) * (((2 * (np.pi ** 2)* (-1)**n) * (1/(n+16) + 1/(n-16) -2/n)) + (((-1)**n) -1)* (8/(n**3) -4/((n+16)**3)-4/((n-16)**3))))
    return Dn_arr

def Energy(N):
    Energyarr = []
    narr = []
    for n in range(1,N+1):
        if n==16:
            Dn = (-(3 * np.pi )/256)
            Bn = ((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2)
            E = np.pi * (n**2) * (Bn **2 + Dn ** 2)
            Energyarr.append(E)
            narr.append(n)
            continue
        Bn = ((4)/(n * np.pi)) * np.sin((n * np.pi)/(2)) * np.sin(n/2)
        Dn = (1/(n * np.pi)) * (((2 * (np.pi ** 2)* (-1)**n) * (1/(n+16) + 1/(n-16) -2/n)) + (((-1)**n) -1)* (8/(n**3) -4/((n+16)**3)-4/((n-16)**3)))
        E = np.pi * (n**2) * (Bn **2 + Dn ** 2)
        Energyarr.append(E)
        narr.append(n)
    return Energyarr, narr

def L2_norm(u_num, u_exact, deltax, deltat):
    diff = u_num - u_exact # no need for for loop since these *should* be equal in size and numpy will handle it.
    return np.sqrt(np.sum(diff**2) * deltax * deltat)

def relative_L2_norm(u_num, u_exact, deltax, deltat):
    '''
    Returns the relative L2 norm
    '''
    diff = u_num - u_exact
    return np.sqrt(np.sum(diff**2) * deltax * deltat) / np.sqrt(np.sum(u_exact**2) * deltax * deltat)

def CN2(tmax,deltat,Nx=101):
    xvals = np.linspace(0,2 * np.pi,Nx)
    deltax = (2*np.pi)/((Nx-1)) # Nx - 1 because it makes numbers nice, and its in notes
    tvals = np.arange(0,tmax,deltat) # + deltat to make sure everything syncs - normally there but removed for the time being
    Nt = len(tvals)
    
    x_in = xvals[1:-1] # defining an interior (not including boundary conditions)
    N_in = len(x_in)
    x,t = np.meshgrid(xvals,tvals)
    
    alpha = (4 * (deltat)**2) / deltax**2
    beta = 2 * (1 - alpha) 

    firstindex, secondindex = indexfinderWave(x_in)
    initial_u_0 = np.zeros(N_in)
    for i in range(0,N_in):
        if i >= firstindex and i <= secondindex:
            initial_u_0[i] = 1

    initial_u_1 = np.zeros(N_in)
    for i in range(0,N_in):
        if i >= firstindex and i <= secondindex:
            initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + 1
        else:
            initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2)

    for i in range(0,N_in):
        if i - 1 == firstindex:
            initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (-1)
            continue
        if i == firstindex:
            initial_u_1[i] = 1 + deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (-1)
            continue
        if i >= firstindex + 2 and i <= (secondindex - 2):
            initial_u_1[i] = 1 + deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (0)
            continue
        if i - 1 == secondindex:
            initial_u_1[i] = 1 + deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (-1)
            continue
        if i == secondindex:
            initial_u_1[i] = 1 + deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (-1)
            continue
        initial_u_1[i] = deltat * (x_in[i]**2) * ((np.sin(4*x_in[i])) **2) + (alpha/2) * (0)

    zeta = alpha/2
    eta = -(alpha+1)
    phi = (alpha - 2)


    P = B(zeta,eta,N_in) # alpha = zeta, beta = eta
    Q = B(-zeta,phi,N_in)
    A = np.linalg.inv(P)
    Z = A @ Q


    u_n0 = np.array(initial_u_0) # making code easier to read
    u_n1 = np.array(initial_u_1)
    U_in = [u_n0.copy(), u_n1.copy()] # doesn't include boundary conditions
    #U.append(u_n.copy())
    for _ in range(Nt-2):
        u_np1 = np.linalg.solve(P, Q @ u_n1 - u_n0)
        U_in.append(u_np1.copy())
        u_n0, u_n1 = u_n1, u_np1
    if deltat > deltax/2:
        print('Δt too large, may be unstable')
    U_full = np.zeros((Nt,Nx))
    U_full[:,1:-1] = np.array(U_in) # adding on boundary condition results i.e. 0 at start of x and end x spectrum (0,2pi)
    return x,t,U_full,deltax

def solve_tridiag(a, b, c, d):
    n = len(b)
    ac = a.copy().astype(float)
    bc = b.copy().astype(float)
    cc = c.copy().astype(float)
    dc = d.copy().astype(float)
    for i in range(1, n):
        m = ac[i-1] / bc[i-1]
        bc[i] -= m * cc[i-1]
        dc[i] -= m * dc[i-1]
    x = np.zeros(n, float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    return x

def CN(tmax, deltat, Nx=101):
    xvals = np.linspace(0, 2*np.pi, Nx)
    deltax = (2*np.pi)/(Nx-1)
    tvals = np.arange(0, tmax, deltat)
    Nt = len(tvals)
    x_in = xvals[1:-1]
    N_in = len(x_in)
    alpha = (4*(deltat**2))/(deltax**2)
    first = int(((np.pi-1)/(2*np.pi))*(Nx-1))
    last = int(((np.pi+1)/(2*np.pi))*(Nx-1)) - 1
    first -= 1
    last -= 1
    u0 = np.zeros(N_in)
    for i in range(N_in):
        if first <= i <= last:
            u0[i] = 1.0
    u1 = np.zeros(N_in)
    for i in range(N_in):
        base = deltat*(x_in[i]**2)*(np.sin(4*x_in[i])**2)
        if first <= i <= last:
            lap = 0
            if i == first:
                lap = -1
            elif i == last:
                lap = -1
            u1[i] = 1 + base + (alpha/2)*lap
        else:
            lap = 0
            u1[i] = base + (alpha/2)*lap
    zeta = alpha/2
    eta = -(alpha+1)
    phi = alpha - 2
    aP = np.full(N_in-1, zeta)
    bP = np.full(N_in, eta)
    cP = np.full(N_in-1, zeta)
    aQ = np.full(N_in-1, -zeta)
    bQ = np.full(N_in, phi)
    cQ = np.full(N_in-1, -zeta)
    U = [u0.copy(), u1.copy()]
    un0 = u0.copy()
    un1 = u1.copy()
    for _ in range(Nt-2):
        rhs = bQ*un1
        rhs[:-1] += aQ*un1[1:]
        rhs[1:] += cQ*un1[:-1]
        rhs += un0
        un2 = solve_tridiag(aP, bP, cP, rhs)
        U.append(un2.copy())
        un0, un1 = un1, un2
    Ufull = np.zeros((Nt, Nx))
    Ufull[:, 1:-1] = np.array(U)
    X, T = np.meshgrid(xvals, tvals)
    return X, T, Ufull, deltax



#endregion
#region ------------------------- Results -------------------------

xaw,taw,uaw,Et= AnalyticalWave(101,5,100)

xnw,tnw,unw,deltax = NumericalWave(5,0.0005)

xpw,tpw,upw = WavePerturb(100,5,0.05,100)

X,T,U,_= CN(5,0.0005)
B = Bn(100)
D = Dn(30)
stepB = range(1, len(B)+1)
stepD = range(1, len(D)+1)
print(relative_L2_norm(U,uaw,deltax,0.0005))
#endregion
#region ------------------------- Plotting -------------------------

fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.plot_surface(xaw,taw,uaw,cmap='viridis', alpha=0.9)

ax.plot_surface(X,T,U,cmap='viridis', alpha=0.9)

ax.set_title('Crank Nicholson Wave Equation')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

ax.view_init(elev=20, azim=-10) # angle of the graph

plt.show()
'''
plt.plot(stepD,D, 'k')
plt.title('Value of $D_n$ against $n$')
plt.ylabel('$D_n$')
plt.xlabel('$n$')
plt.scatter(16,(-(3 * np.pi )/256), label = '$D_{16}$',color = 'red')
plt.ylim(-3,3)


plt.plot(stepB,B,'k')
plt.title('Value of $B_n$ against $n$')
plt.ylabel('$B_n$')
plt.xlabel('$n$')



#E, n  = Energy(100000)

plt.plot(n,E, 'k')
plt.title('Energy of Wave Equation against $n$')
plt.ylabel('$E(t)$')
plt.xlabel('$n$')

plt.legend()
plt.grid(True)
plt.minorticks_on()
plt.show()
'''
#endregion

































#endregion