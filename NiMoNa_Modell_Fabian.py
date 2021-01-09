import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib as mpl
import networkx as nx

#*******************************************************
#Topologies
def circle(N):
    A = np.zeros((N, N))
    for i in range(N-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    A[N-1, 0] = 1
    A[0, N-1] = 1
    return A

def star(N):
    A = np.zeros((N, N))
    A[0] = np.ones(N)
    for i in range(N):
        A[i][0] = 1
        A[0][i] = 1
    A[0][0] = 0
    return A

def wheel(N):
    A = np.zeros((N, N))
    for i in range(N-2):
        A[i, i+1] = 1
        A[i+1, i] = 1
        A[i, N-1] = 1
        A[N-1, i] = 1
    A[N-2, N-1] = 1
    A[N-1, N-2] = 1
    A[N-2, 0] = 1
    A[0, N-2] = 1
    return A

def mesh(N):
    A = np.ones((N, N))
    for i in range(N):
        A[i, i] = 0
    return A

#*******************************************************
#Matrix transformation functions

def enlarge_matrix(A):
    #takes matrix of any shape as input and returns adj. matrix with one row and one columns of zeros added
    B = np.ones([0,A.shape[1]+1])
    for row in A:
        B = np.append(B, [np.append(row, 0)], 0)
    B = np.append(B, [np.zeros(A.shape[1]+1)], 0)
    return B

def enlarge_matrix_N(A, n):
    #enlarge given matrix by n rows and columns of zeros
    for i in range(n):
        A = enlarge_matrix(A)
    return A

def make_connection(A, id1, id2):
    A[id1][id2] = 1
    A[id2][id1] = 1

def delete_connection(A, id1, id2):
    A[id1][id2] = 0
    A[id2][id1] = 0

def add_followers(A, i, n):
    #adds n followers (star topology) of agent i to adj. matrix A and returns new adj. matrix
    old_n = A.shape[1]
    A = enlarge_matrix_N(A, n)
    for j in range(n):
        make_connection(A, i, old_n+j)
    return A

def influencer_network(N, m):
    #create network of n connected agents with m followers,
    #where m is of type int (adding m followers to each agent) or list of len(N) of ints (adding m[i] followers to agent i)
    A = mesh(N)
    for i in range(N):
        if isinstance(m, int):
            A = add_followers(A, i, m)
        else:
            A = add_followers(A, i, m[i])            
    return A

#*******************************************************


#runge-kutta-4 function
def rk4(f, t, x, h, par):
    k1 = f(t,x,par)
    k2 = f(t + 0.5*h, x + 0.5*h*k1,par)
    k3 = f(t + 0.5*h, x + 0.5*h*k2,par)
    k4 = f(t + h, x + h*k3,par)
    return (h*(k1 + 2*k2 + 2*k3 + k4)/6)

#right hand side of ODE
def rhs(t,x,par):
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    mat_a = par[5]
    sum_vec = np.matmul(mat_a, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        #if d, u or b are vectors of length len(x), add [i] to line below
        result[i] = -d*x[i]+u*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b
    return result

def plot(f, par, *N):
    print(f"Now plotting {f.__name__}.")    
    if f.__name__ == "influencer_network":
        A = f(N[0], N[1])
        if isinstance(N[1], int):
            m = N[0]+N[0]*N[1]
        else:
            m = N[0]+sum(N[1])
    else:
        m = N[0]
        A = f(m)
    # Initialize N nodes randomly, with opinions centered around 0
    xs = (np.random.uniform(size=(m))-0.5)*2
    par.append(A)
    #define other parameters:
    t_span = [0,7]
    t, x = t_span[0], xs
    h = 5e-1
    # now get plot-positions
    rows, cols = np.where(A == 1.)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    plot_positions = nx.drawing.spring_layout(gr)  
    # and create a colorbar
    vmin = -1
    vmax = 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('coolwarm')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure()
    camera = Camera(fig)
    while t <= t_span[1]:
        nx.draw(gr, pos=plot_positions, node_size=500, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax)
        camera.snap()
        t, x = t + h, x + rk4(rhs, t, x, h, par)
    plt.colorbar(sm)
    animation = camera.animate()
    animation.save(f'{f.__name__}.gif', writer='PillowWriter', fps=3)
    par.pop()
    print(f"Saved {f.__name__}.gif")


#Initial Data
# Number of nodes
N = 13
#parameters in order d, u, alpha, gamma, b
params = [1,0.31,1.2,-1.3,0]


plot(wheel, params, N)
plot(star, params, N)
plot(circle, params, N)
plot(mesh, params, N)
N = 3
m = [4,5,6]
plot(influencer_network, params, N, m)