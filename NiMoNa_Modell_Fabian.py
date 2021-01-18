# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 21:56:08 2021
@author: Fabian Bremer
"""

import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import matplotlib as mpl
import networkx as nx

#*******************************************************
#Matrix transformation functions

def enlarge_matrix(A):
    """
    takes matrix of any shape as input and returns adj. matrix with one row 
    and one columns of zeros added

    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix

    Returns
    -------
    B : np.array (matrix) - enlarged adjacency matrix

    """
    B = np.zeros([A.shape[0]+1, A.shape[1]+1])
    B[:-1, :-1] = A
    return B

def enlarge_matrix_N(A, n):
    """
    enlarges given matrix by n rows and columns of zeros

    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix
    n : integer - number of rows and columns A will be enlarged by

    Returns
    -------
    A : np.array (matrix) - enlarged adjacency matrix

    """
    for i in range(n):
        A = enlarge_matrix(A)
    return A

def make_connection(A, id1, id2):
    """
    makes a connection betwween two agents in an adjacency matrix

    Parameters
    ----------
    A :   np.array (matrix) - input adjacency matrix
    id1 : integer - ID of one of the agents in A that is going to be connected
    id2 : integer - ID of the other agent in A that is going to be connected

    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A[id1][id2] = 1
    A[id2][id1] = 1
    return A

def delete_connection(A, id1, id2):
    
    """
    deletes a connection betwween two agents in an adjacency matrix

    Parameters
    ----------
    A :     np.array (matrix) - input adjacency matrix
    id1 :   integer - ID of one of the agents in A that is going to be 
            disconnected
    id2 :   integer - ID of the other agent in A that is going to be
            disconnected

    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix
    """
    A[id1][id2] = 0
    A[id2][id1] = 0
    return A

#*******************************************************
#simple topologies

def circle(N):
    """
    Creates NxN adjacency matrix for circle topology

    Parameters
    ----------
    N : integer - number of agents in network
    
    Returns
    -------
    A : np.array (matrix) - adjacency matrix for circle topology
    """
    A = np.zeros((N, N))
    for i in range(N-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    A[N-1, 0] = 1
    A[0, N-1] = 1
    return A

def star(N):
    """
    Creates NxN adjacency matrix for star topology

    Parameters
    ----------
    N : integer - number of agents in network
    
    Returns
    -------
    A : np.array (matrix) - adjacency matrix for star topology
    """
    A = np.zeros((N, N))
    A[0] = np.ones(N)
    for i in range(N):
        A[i][0] = 1
        A[0][i] = 1
    A[0][0] = 0
    return A

def wheel(N):
    """
    Creates NxN adjacency matrix for wheel topology

    Parameters
    ----------
    N : integer - number of agents in network
    
    Returns
    -------
    A : np.array (matrix) - adjacency matrix for wheel topology
    """
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
    """
    Creates NxN adjacency matrix for mesh topology

    Parameters
    ----------
    N : integer - number of agents in network
    
    Returns
    -------
    A : np.array (matrix) - adjacency matrix for mesh topology
    """
    A = np.ones((N, N))
    for i in range(N):
        A[i, i] = 0
    return A

#*******************************************************
#a little more complicated networks / topologies and functions to create them
#*************
#Influencer Network
def add_followers(A, i, n):
    """
    adds n followers (star topology) of agent i to adj. matrix A and 
    returns new adj. matrix

    Parameters
    ----------
    A : np.array (matrix) - input adjacency matrix
    i : integer - ID of agent in network that followers shall be added to
    n : integer - number of followers to be added

    Returns
    -------
    A :  np.array (matrix) - adjacency matrix
    """
    old_n = A.shape[1]
    A = enlarge_matrix_N(A, n)
    for j in range(n):
        A = make_connection(A, i, old_n+j)
    return A

def influencer_network(N, m):
    """
    Create network of N connected agents with m followers, where m is of type 
    int (adding m followers to each agent) or list of len(N) of ints (adding 
    m[i] followers to agent i)

    Parameters
    ----------
    N : Integer - Number of influencers
    m : Integer or List of Integers - Number of followers for each influencer

    Returns
    -------
    A : np.array (matrix) - adjacency matrix
    """
    A = mesh(N)
    for i in range(N):
        if isinstance(m, int):
            A = add_followers(A, i, m)
        else:
            A = add_followers(A, i, m[i])            
    return A

#*************
# Random Tree Topology
def add_rand_branches(A, low, high):
    """
    Adds a (random) number of branches to an existing tree. New branches will 
    only be added to a node, if that node is only connected with one other 
    node (so if it is on the outside of the tree). The random number can be 
    bounded and therefore also fixed on a not random int by setting low=high

    Parameters
    ----------
    A :   Adjacency matrix of network that the branches shall be added to
    low : lowest possible integer for the random choice of added branches to 
          each node
    high :highest possible integer for the random choice of added branches to 
          each node 
          
          NOTE: if low = high = n, then n branches will be added to each node
    Returns
    -------
    A : np.array (matrix) - changed adjacency matrix

    """
    A_copy = A
    counter = 0
    for row in A_copy:
        if np.sum(row) < 2:
            n = np.random.randint(low=low, high=high+1)
            A = add_followers(A, counter, n)
        counter += 1
    return A

def tree(depth, low, high):
    """
    Creates a tree topology with the "depth" being the number of nodes 
    counting from the central node and going the longest path to end of tree
    (length of all paths is identical if low >0)

    Parameters
    ----------
    depth : integer, depth as defined above 
    low :   integer "low" that is passed to add_rand_branches function (see
            docstring there)
    high :  integer "high" that is passed to add_rand_branches function (see
            docstring there)

    Returns
    -------
    A : np.array (matrix) - adjacency matrix

    """
    A = np.zeros([1, 1])
    for _ in range(depth):
        A = add_rand_branches(A, low, high)
    return A

#*******************************************************
#solvers

def k_i(i,f,t,x,h,a,b,par):
    """
    returns k_i function for Runge-Kutta-methods 

    Parameters
    ----------
    i : integer
        determines which k_i is to be calculated.
    f : callable
        the right hand side of the ODE that needs to be solved.
        It has format f(t,x) where x can be a list of any length
    t : float
        time-variable of f(t,x).
    x : float or list of floats
        x-varibale of f(t,x).
    h : float
        stepsize of timestep.
    a : np.array (list)
        vector of nodes in Butcher-tableau for RK-methods
    b : np.array (matrix)
        Runge-Kutta-matrix in Butcher tableau
    par : list
        parameters needed for f.

    Returns
    -------
    float or list of floats, depending on f(t,x)
        k_i function for Runge-Kutta-methods 

    """
    real_i = i-1
    k_sum = 0
    for j in range(real_i):
        k_sum += b[real_i][j]*k_i(j+1,f,t,x,h,a,b,par)
    return f(t + h*a[real_i], x + h*k_sum, par)

#runge-kutta (4)
def rk4(f, t, x, h, par):
    """
    calculates one step for the classic Runge-Kutta method (RK4)

    Parameters
    ----------
    f :   callable - the right hand side of the ODE that needs to be solved.
          It has format f(t,x) where x can be a list of any length
    t :   float - time-variable of f(t,x)
    x :   float or list of floats - x-varibale of f(t,x)
    h :   float - stepsize of timestep
    par : list - other parameters for f

    Returns
    -------
    h*ck_sum : for the step x_n+1 = x + h*ck_sum (one step for x in RK4)
    """
    a = np.array([0, 0.5, 0.5, 1])
    b = np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]])
    c = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    ck_sum = 0
    for i in range(len(c)):
        ck_sum += c[i]*k_i(i+1, f, t, x, h, a, b, par)
    return h*ck_sum

#count discarded steps in Fehlberg method
dis_steps = 0

#runge-kutta-fehlberg (4/5)
def rkf45(f, t, x, h, par):
    """
    calculates one step for the Runge-Kutta-Fehlberg method (RKF 45) while 
    counting discarded steps

    Parameters
    ----------
    f :   callable - the right hand side of the ODE that needs to be solved.
          It has format f(t,x) where x can be a vector / a list of any length
    t :   float - time-variable of f(t,x)
    x :   float or list of floats - x-varibale of f(t,x)
    h :   float - stepsize of timestep
    par : list - other parameters for f

    Returns
    -------
    h*ck_sum : for the step x_n+1 = x + h*ck_sum (one step for x in RK4)
    h_new    : new stepsize for next step
    """
    global dis_steps
    eps_tol = 1e-7
    safety = 0.9
    a = np.array([0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0])
    b = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    		[1.0/4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    		[3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
    		[1932.0/2197.0, (-7200.0)/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
    		[439.0/216.0, -8.0, 3680.0/513.0, (-845.0)/4104.0, 0.0, 0.0],
    		[(-8.0)/27.0, 2.0, (-3544.0)/2565.0, 1859.0/4104.0, (-11.0)/40.0, 0.0]])
    c = np.array([25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, (-1.0)/5.0, 0.0]) # coefficients for 4th order method
    c_star = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, (-9.0)/50.0, 2.0/55.0]) # coefficients for 5th order method
    cerr = c - c_star
    err = 0
    k = np.ones([0,len(x)])
    for i in range(6):
        k = np.append(k, [k_i(i+1, f, t, x, h, a, b, par)], 0)
        err += cerr[i]*k[i]
    err = h*np.absolute(err).max() 
    if err > eps_tol:
        dis_steps += 1
        h_new = safety * h * (eps_tol/err)**0.25
        return rkf45(f, t, x, h_new, par)
    else:
        ck_sum = 0
        h_new = safety * h * (eps_tol/err)**0.2
        for i in range(len(c_star)):
            ck_sum += c_star[i]*k[i]
        return h * ck_sum, h_new

#*******************************************************
#right hand side of ODE
def rhs(t,x,par):
    """
    right hand side of ODE - see https://arxiv.org/abs/2009.13600

    Parameters
    ----------
    t :   not used in this particular ODE, kept for comparability 
    x :   np.array - list of states of opinion of the agents in the network
    par : list - list of parameters used in this model, which are:
          d -     > 0, resistance to becoming opinionated for each agent
          u -     > 0, attention to social influence of each agent
          alpha - > 0, self-reinforcement of opinion
          gamma - cooperative agents (gamma > 0) give rise to agreement, while
                  competitive agents (gamma < 0) give rise to disagreement
          b -     bias or additive input
          A -     adjacency matrix representing network structure
    Returns
    -------
    result : np.array - right hand side of ODE
    """
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        #if d, u or b are vectors of length len(x), add [i] to line below
        result[i] = -d*x[i]+u*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b
    return result

#*******************************************************
#Plotting

def plot(f, par, *N):
    """
    Plotting function for this particular problem

    Parameters
    ----------
    f :   callable - function that determines network topology
    par : list - list of parameters needed by rhs of ODE
    *N :  information about size of network topology, needed by f
    
    Returns
    -------
    None.

    """
    global dis_steps
    used_steps = 0
    print(f"Now plotting {f.__name__}.")    
    if f.__name__ == "influencer_network":
        A = f(N[0], N[1])
    elif f.__name__ == "tree":
        A = f(N[0], N[1], N[2])
    else:
        A = f(N[0])
    m = A.shape[0]
    # Initialize m nodes randomly, with opinions centered around 0
    xs = (np.random.uniform(size=(m))-0.5)*2
    par.append(A)
    #define other parameters:
    t_null = 0
    t, x = t_null, xs
    h = 0.01 #initial stepsize
    h_max = 1.1 # for higher h_max the plots stop converging, esp. mesh
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
    max_steps = 500 #max number of steps in solver method, that will be taken
    while h <= h_max:
        nx.draw(gr, pos=plot_positions, node_size=500, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax)
        camera.snap()
        rk_step_x, h_new = rkf45(rhs, t, x, h, par)
        t, x = t + h, x + rk_step_x
        h = h_new
        used_steps += 1
        if used_steps > max_steps:
            print(f"Stopped plotting because more than {max_steps} steps were used. This should not happen.")
            break
    plt.colorbar(sm)
    animation = camera.animate()
    animation.save(f'{f.__name__}.gif', writer='PillowWriter', fps=10)
    par.pop()
    print(f"Discarded {dis_steps} steps. Used {used_steps} steps.")
    dis_steps = 0
    print(f"Saved {f.__name__}.gif")
    return None


#*******************************************************
#Initial Data
# Number of agents
N = 13
#parameters in order d, u, alpha, gamma, b
params = [1,0.31,1.2,-1.3,0]


#*******************************************************
#Let the fun begin
#plot(wheel, params, N)
#plot(star, params, N)
#plot(circle, params, N)
#plot(mesh, params, N)
#change N and introduce m for plotting influencer network
N = 3
m = [4,5,6]
#plot(influencer_network, params, N, m)
#introduce low and high for tree network. See explanation in tree function above.
low = 2
high = 2
plot(tree, params, N, low, high)