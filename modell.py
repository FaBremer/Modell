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

def everybody_connected(N):
    A = np.ones((N, N))
    for i in range(N):
        A[i, i] = 0
    return A

#*******************************************************
#Initial Data
# Number of nodes
N = 13
# Initialize N nodes randomly, with opinions centered around 0
xs = (np.random.uniform(size=N)-0.5)*2


#def rk4(f):
#    return lambda t, x, h: (lambda k1: (lambda k2: (lambda k3: (lambda k4: (k1 + 2*k2 + 2*k3 + k4)/6)( h * f( t + h  , x + k3 )))( h * f( t + h/2, x + k2/2 )))( h * f( t + h/2, x + k1/2)))( h * f( t, x) )
def rk4a(f, t, x, h, par):
    k1 = f(t,x,par)
    k2 = f(t + 0.5*h, x + 0.5*h*k1,par)
    k3 = f(t + 0.5*h, x + 0.5*h*k2,par)
    k4 = f(t + h, x + h*k3,par)
    return (h*(k1 + 2*k2 + 2*k3 + k4)/6)
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

def plot(f, N):
    A = f(N)
    pars = [1,0.31,1.2,-1.3,0,A]
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
        t, x = t + h, x + rk4a(rhs, t, x, h, pars)
    plt.colorbar(sm)
    animation = camera.animate()
    animation.save(f'{f.__name__}.gif', writer='PillowWriter', fps=3)

plot(wheel, N)
plot(star, N)
plot(circle, N)
plot(everybody_connected, N)