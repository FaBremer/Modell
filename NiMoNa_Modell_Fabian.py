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
import random


def decision(probability):
    """
    returns Returns TRUE with a probability of input, FALSE otherwise

    Parameters
    ----------
    probability : float
        Should be in interval [0,1] and represents probability for the
        function to return TRUE.

    Returns
    -------
    boolean
        True with probability probability, False otherwise.

    """
    return random.random() < probability

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
    A[0, N-1] = 1
    A[N-1, 0] = 1
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

def connect_networks(*A):
    """
    connects different networks together via their first agent

    Parameters
    ----------
    *A : tuple of np.arrays (matrices)
        networks to be linked together.

    Returns
    -------
    B : np.array (matrix)
        connected network.

    """
    
    n_list = []
    inf_IDs = [0]
    for net in A:
        n_list.append(net.shape[0])
        inf_IDs.append(np.sum(n_list))
    inf_IDs.pop()
    n = np.sum(n_list)
    B = np.zeros([n,n])
    k = 0
    for i in range(len(A)):
        B[k:k+n_list[i], k:k+n_list[i]] = A[i]
        k += n_list[i]
    for inf_ID in inf_IDs:
        for j in range(len(inf_IDs)):
            if inf_IDs[j] == inf_ID:
                continue
            else:
                B = make_connection(B, inf_IDs[j], inf_ID)
    return B, inf_IDs

def small_world_network(N, k=2):
    """
    Creates small world network as in https://www.nature.com/articles/30918
    Parameters
    ----------
    N : integer
        Number of nodes.
    k : integer
        Number of neighbours to which each node is connected on each side.
        The default is 2.
    Returns
    -------
    A : np.array (matrix)
        Adjacency matrix for such a network.
    """
    
    A = circle(N)
    for i in range(N):
        for j in range(k-1):
            A = make_connection(A, i, i-(j+2))
    return A

def small_world_network_with_rand(N, p, k=2):
    """
    Creates small world network as in https://www.nature.com/articles/30918
    
    Parameters
    ----------
    N : integer
        Number of nodes.
    p : float
        propability for randomization (see paper above)
    k : integer
        Number of neighbours to which each node is initially connected on each 
        side. The default is 2.

    Returns
    -------
    A : np.array (matrix)
        Adjacency matrix for such a network.

    """
    B = small_world_network(N, k)
    for j in range(k):
        for i in range(N):
            rand_agent = random.randint(0, N-1)
            while ((rand_agent == i) or (B[rand_agent][i] == 1)):
                rand_agent = random.randint(0, N-1)
            if decision(p):
                B = delete_connection(B, i-(j+1), i)
                B = make_connection(B, rand_agent, i)
    return B

#*******************************************************
#solvers

def k_i2(i,f,t,x,h,a,b,par):
    """
    The better version of k_i (see below). To see difference in implementation,
    compare rk4, where k_i is implemented and rkf45, where k_i2 is implemented.
    Instead of rk4 and r_i the betteer versions rkf45

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
    k_list : list
        list of k_i functions for Runge-Kutta-methdos.
    """
    k_list = [] 
    for j in range(i):
        a_j = a[j]
        b_j = b[j]
        res = k_step(f, t, x, h, a_j, b_j, par, k_list)
        k_list.append(res)
        #k_sum += b[i-1][j]*k_i(j+1,f,t,x,h,a,b,par)
    return k_list

def k_step(f,t,x,h,a,b,par,k):
    """
    calculates next k-function for Runge-Kutta-methods, given the already 
    calculated k-functions

    Parameters
    ----------
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
    b : np.array (list)
        row of Runge-Kutta-matrix in Butcher tableau
    par : list
        parameters needed for f.
    k : list
        list of previously calculated k-functions.

    Returns
    -------
    float or list of floats, depending on f(t,x)
        k_i function for Runge-Kutta-methods 
    """
    bk_sum = 0
    for i in range(len(k)):
        bk_sum += b[i]*k[i]
    return f(t+h*a, x+h*bk_sum, par)

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

#runge-kutta (4) - not used in code, left for documentation. For better version
#use Runge-Kutta-Fehlberg method (see rkf45 below)
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
    k = k_i2(len(cerr), f, t, x, h, a, b, par) #
    for i in range(6):
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
    if f.__name__ == "influencer_network" or f.__name__ == "small_world_network_with_rand":
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
    if f.__name__ == "small_world_network_with_rand":
        plot_positions = nx.circular_layout(gr)
        print(plot_positions)
        new_dict = dict(zip(sorted(gr.nodes()),plot_positions.values()))
        plot_positions = new_dict
    else:
        plot_positions = nx.spring_layout(gr)
    # and create a colorbar
    gr.add_edge(0, N[0]-1)
    vmin = -1
    vmax = 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('coolwarm')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig = plt.figure()
    camera = Camera(fig)
    max_steps = 500 #max number of steps in solver method, that will be taken
    """
    labels={}
    for i in range(m):
        labels[i] = f"{i}"
    """
    while h <= h_max:
        nx.draw(gr, pos=plot_positions, node_size=500, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax)
        #nx.draw_networkx_labels(gr, plot_positions, labels)
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
#Plotting simple networks
#Initial Data
# Number of agents
#N = 13
#parameters in order d, u, alpha, gamma, b
#params = [1,0.31,1.2,-1.3,0]
#***********
#Let the fun begin
#plot(wheel, params, N)
#plot(star, params, N)
#plot(circle, params, N)
#plot(mesh, params, N)
#change N and introduce m for plotting influencer network
#N = 3
#m = [4,5,6]
#plot(influencer_network, params, N, m)
#introduce low and high for tree network. See explanation in tree function above.
#low = 2
#high = 2
#plot(tree, params, N, low, high)

N = 20
p = 0.8
params = [1,0.31,1.2,-1.3,0]

#plot(small_world_network, params, N)
#plot(circle, params, 3)
#plot(small_world_network_with_rand, params, N, p)

"""
NOW: PLOTTING REALISTIC INFLUENCER NETWORK (RIN)
"""
#change old plot function to plot realistic influencer network (rin)

def plot_rin(f, par, *N):
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
    print(f"Now plotting realistic_{f.__name__}.")    
    A = f(N[0], N[1])
    m = A.shape[0]
    if len(N) > 2:
        node_size = N[2]
    else:
        node_size = 500
    # Initialize m nodes randomly, with opinions centered around 0
    xs = (np.random.uniform(size=(m))-0.5)*2
    #change opinion of influencers to strong opinion
    for i in range(N[0]):
        if i >= N[0]//2:
            xs[i] = 2
        else:
            xs[i] = -2
    par.append(A)
    #define other parameters:
    t_null = 0
    t, x = t_null, xs
    h = 0.01 #initial stepsize
    h_max = 5 
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
        nx.draw(gr, pos=plot_positions, node_size=node_size, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax)
        camera.snap()
        rk_step_x, h_new = rkf45(rhs_rin, t, x, h, par)
        t, x = t + h, x + rk_step_x
        h = h_new
        used_steps += 1
        if used_steps > max_steps:
            print(f"Stopped plotting because more than {max_steps} steps were used. This should not happen.")
            break
    plt.colorbar(sm)
    animation = camera.animate()
    animation.save(f'realistic_{f.__name__}.gif', writer='PillowWriter', fps=10)
    par.pop()
    print(f"Discarded {dis_steps} steps. Used {used_steps} steps.")
    dis_steps = 0
    print(f"Saved realistic_{f.__name__}.gif")
    return None

def plot_rin_2(A, f, xs, par, *N):
    global dis_steps
    used_steps = 0
    print(f"Now plotting realistic_{f.__name__}.")    
    node_size = N[2]
    par.append(A)
    #define other parameters:
    t_null = 0
    t, x = t_null, xs
    h = 0.01 #initial stepsize
    h_max = 5 
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
        nx.draw(gr, pos=plot_positions, node_size=node_size, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax)
        camera.snap()
        rk_step_x, h_new = rkf45(rhs_rin, t, x, h, par)
        t, x = t + h, x + rk_step_x
        h = h_new
        used_steps += 1
        if used_steps > max_steps:
            print(f"Stopped plotting because more than {max_steps} steps were used. This should not happen.")
            break
    plt.colorbar(sm)
    animation = camera.animate()
    animation.save(f'realistic_{f.__name__}.gif', writer='PillowWriter', fps=10)
    par.pop()
    print(f"Discarded {dis_steps} steps. Used {used_steps} steps.")
    dis_steps = 0
    print(f"Saved realistic_{f.__name__}.gif")
    return None
def rhs_rin(t,x,par):
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
        result[i] = -d[i]*x[i]+u[i]*np.tanh(alpha[i]*x[i]+gamma[i]*sum_vec[i])+b
    return result

N = 3
m = [60,50,70]

H = influencer_network(1, m[0])
I = influencer_network(1, m[1])
J = influencer_network(1, m[2])

A, ids = connect_networks(H,I,J)

matrix_size = A.shape[0]

d_inf, d_fol = 0.15, 0.3
my_d = np.zeros(matrix_size)
my_d[:] = d_fol

u_inf, u_fol = 0.01, 0.35
my_u = np.zeros(matrix_size)
my_u[:] = u_fol
    
alpha_fol =  0.3
my_alpha = np.zeros(matrix_size)
my_alpha[:] = alpha_fol
    
gamma_inf, gamma_fol = 1.5, 1.5
my_gamma = np.zeros(matrix_size)
my_gamma[:] = gamma_fol

node_size = np.ones(N+np.sum(m))
node_size[:] = 50

# Initialize m nodes randomly, with opinions centered around 0
xs = (np.random.uniform(size=(matrix_size))-0.5)*0.2


node_size[0] = 300
my_d[0] = d_inf
my_u[0] = u_inf
my_gamma[0] = gamma_inf
xs[0] = -3
my_alpha[0] = 10 * (m[0]/np.sum(m))
for h in range(1,len(m)):
    node_size[m[0]+h] = 300
    xs[m[0]+h] = 3
    my_d[m[0]+h] = d_inf
    my_gamma[m[0]+h] = gamma_inf
    my_u[m[0]+h] = u_inf
    my_alpha[m[0]+h] = 5 * (m[h]/np.sum(m))

params = [my_d,my_u,my_alpha,my_gamma,0]

plot_rin_2(A, influencer_network, xs, params, N, m, node_size)





"""

DISCARDED PART OF PROJECT:
    
    The following part has been discarded, because it generates "correct" 
    results by wrongly twisting the modellparameters. To emulate realistic
    behaviour of influencer networks mainly the parameter b - bias - was 
    changed. However in the model this represents the additive input of
    influences outside the model, not in the model itself as is used here.
    This discarded part is kept here for transparency reasons.

#*******************************************************
#Plotting more realistic influencer networks (rin)
#edited rhs and plotfunction:
#NOTE: various variations of the rhs will be used in the following. No further
#docstrings will be provided. For documentation, see docstring of rhs() above

def plot_rin(f, f_2, par, *N):

    Plotting function for realistic influencer network.

    Parameters
    ----------
    f :   callable - function that determines network topology
    f_2:  callable - rhs that is going to be used
    par : list - list of parameters needed by rhs of ODE
    *N :  information about size of network topology, needed by f
    
    Returns
    -------
    None.

    global dis_steps
    global plot_counter
    global node_size
    used_steps = 0
    print(f"Now plotting realistic influencer network no. {plot_counter+1}.")    
    A = f(N[0], N[1])
    # Initialize all nodes randomly, with opinions centered around 0
    xs = (np.random.uniform(size=(A.shape[0]))-0.5)*2
    #change opinion of influencers to strong opinion
    for i in range(N[0]):
        if i >= N[0]//2:
            xs[i] = 2
        else:
            xs[i] = -2
    par.append(A)
    #define other parameters:
    t_null = 0
    t, x = t_null, xs
    h = 0.01 #initial stepsize
    h_max = 3
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
        nx.draw(gr, pos=plot_positions, node_size=node_size, node_color=x, cmap='coolwarm', vmin=vmin, vmax=vmax) #ADJUST NODE SIZE HIERE FOR LARGER NETWORKS
        camera.snap()
        rk_step_x, h_new = rkf45(f_2, t, x, h, par)
        t, x = t + h, x + rk_step_x
        h = h_new
        used_steps += 1
        if used_steps > max_steps:
            print(f"Stopped plotting because more than {max_steps} steps were used. This should not happen.")
            break
    plt.colorbar(sm)
    animation = camera.animate()
    plot_counter += 1
    animation.save(f'realistic_{f.__name__}_{plot_counter}.gif', writer='PillowWriter', fps=10)
    par.pop()
    print(f"Discarded {dis_steps} steps. Used {used_steps} steps.")
    dis_steps = 0
    print(f"Saved realistic_{f.__name__}_{plot_counter}.gif")
    return None


plot_counter = 0 #for filenaming purposes
#first plot
#use old rhs
def rhs_rin_1(t,x,par):
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b
    return result

node_size = 30
#start out with 2 Influencers and 100 followers each
N_1 = 2
m_1 = 100
params = [1,0.31,1.2,1.3,0]
plot_rin(influencer_network, rhs_rin_1, params, N_1, m_1)

wait = input("Press Enter to continue.")

#second plot
#NOW: give rhs a list of u_is instead of a single u as in: make the social
#influence different for followers and influencers
def rhs_rin_2(t,x,par):
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u[i]*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b
    return result

node_size = 30
N_2 = 2
m_2 = 100
u_inf = 0.001
u_fol = 0.7
my_u = np.zeros(influencer_network(N_2, m_2).shape[0])
my_u[:N_2] = u_inf
my_u[N_2:] = u_fol
params = [0.1,my_u,1.2,1.3,0]
plot_rin(influencer_network, rhs_rin_2, params, N_2, m_2)

wait = input("Press Enter to continue.")

#third plot
#NOW: give rhs a list of b_is instead of a single b as in: make the additive 
#input (bias) different for followers and influencers
def rhs_rin_3(t,x,par):
    global N_3
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    for j in range(N_3):
        b[j] = x[j]
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u[i]*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b[i]
    return result

node_size = 30
N_3 = 2
m_3 = 100
u_inf = 0.001
u_fol = 0.7
my_u = np.zeros(influencer_network(N_3, m_3).shape[0])
my_u[:N_3] = u_inf
my_u[N_3:] = u_fol
my_b = np.zeros(influencer_network(N_3, m_3).shape[0])

params = [1,my_u,1.2,1.3,my_b]
plot_rin(influencer_network, rhs_rin_3, params, N_3, m_3)

wait = input("Press Enter to continue.")

#fourth plot
#NOW: make that bias a little more realistic as in: the influencers interact

def rhs_rin_4(t,x,par):
    global N_4
    global infl_perc
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    b[:N_4] = (np.sum(x[:N_4])/N_4)*infl_perc
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u[i]*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b[i]
    return result

node_size = 30
infl_perc = 0.6
N_4 = 5
m_4 = 50
u_inf = 0.1
u_fol = 0.7
my_u = np.zeros(influencer_network(N_4, m_4).shape[0])
my_u[:N_4] = u_inf
my_u[N_4:] = u_fol
my_b = np.zeros(influencer_network(N_4, m_4).shape[0])

params = [1,my_u,1.2,1.3,my_b]
plot_rin(influencer_network, rhs_rin_4, params, N_4, m_4)

infl_perc = 0.9
plot_rin(influencer_network, rhs_rin_4, params, N_4, m_4)

wait = input("Press Enter to continue.")

#fifth plot
#NOW: change bias of followers

def rhs_rin_5(t,x,par):
    global N_5
    global infl_perc
    global fol_perc
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    b[:N_5] = (np.sum(x[:N_5])/N_5)*infl_perc
    for j in range(N_5):
        b[N_5+j*((len(b)-N_5)//N_5):] = x[j]*fol_perc
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u[i]*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b[i]
    return result

node_size = 30
infl_perc = 0.7
fol_perc = 0.2
u_inf = 0.1
u_fol = 0.35
N_5 = 3
m_5 = 100
my_u = np.zeros(influencer_network(N_5, m_5).shape[0])
my_u[:N_5] = u_inf
my_u[N_5:] = u_fol
my_b = np.zeros(influencer_network(N_5, m_5).shape[0])

params = [1,my_u,1.2,1.3,my_b]

plot_rin(influencer_network, rhs_rin_5, params, N_5, m_5)
wait = input("Press Enter to continue.")


#sixth plot
#NOW: let influencers be aware of no. of followers of other influencers

def rhs_rin_6(t,x,par):
    global N_6
    global m_6
    global infl_perc
    global fol_perc
    d = par[0]
    u = par[1]
    alpha = par[2]
    gamma = par[3]
    b = par[4]
    k=0
    b[:N_6] = (np.sum((x[:N_6]*m_6))/(np.sum(m_6)))*infl_perc
    for j in range(N_6):
        b[N_6+k:] = x[j]*fol_perc
        k = np.sum(m_6[:j+1])
    A = par[5]
    sum_vec = np.matmul(A, x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = -d*x[i]+u[i]*np.tanh(alpha*x[i]+gamma*sum_vec[i])+b[i]
    return result

node_size = 30
infl_perc = 0.7
fol_perc = 0.2
u_inf = 0.1
u_fol = 0.35
N_6 = 3
m_6 = [100,50,10]
my_u = np.zeros(influencer_network(N_6, m_6).shape[0])
my_u[:N_6] = u_inf
my_u[N_6:] = u_fol
my_b = np.zeros(influencer_network(N_6, m_6).shape[0])

params = [1,my_u,1.2,1.3,my_b]
plot_rin(influencer_network, rhs_rin_6, params, N_6, m_6)

"""