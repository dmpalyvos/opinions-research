# -*- coding: utf-8 -*-
# pylint: disable=E1101
'''
Models of Opinion Formation
'''


from __future__ import division, print_function

import numpy as np
import numpy.random as rand
import random as stdrand
import scipy.sparse as sparse
from numpy.linalg import norm, inv

from datetime import datetime
from tqdm import trange

from util import row_stochastic, save_data


def preprocessArgs(s, max_rounds):
    '''Argument processing common for most models.

    Returns:
        N, z, max_rounds
    '''

    N = np.size(s)
    max_rounds = int(max_rounds) + 1  # Round 0 contains the initial opinions
    z = s.copy()

    return N, z, max_rounds


def deGroot(A, s, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the DeGroot Model.

    Runs a maximum of max_rounds rounds of the DeGroot model. If the model
    converges sooner, the function returns.

    Args:
        A (NxN numpy array): Adjacency matrix

        s (1xN numpy array): Initial opinions vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        z = A.dot(z)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('DeGroot converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'dg' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def friedkinJohnsen(A, s, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the Friedkin-Johnsen (Kleinberg) Model.

    Runs a maximum of max_rounds rounds of the Friedkin-Jonsen model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    B = np.diag(np.diag(A))  # Stubborness matrix of the model
    A_model = A - B  # Adjacency matrix of the model

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = z

    for t in trange(1, max_rounds):
        z = A_model.dot(z) + B.dot(s)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Friedkin-Johnsen converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'fj' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def rchoice(weights, nonzero_ids):
    '''Makes a (weighted) random choice.

    Given a vector of probabilities with a total sum of 1, this function
    returns the index of one element of the list with probability equal to
    this element's value. For example, given the vector [0.2, 0.5, 0.3], the
    probability that the function returns 0 is 20%, the probability that
    the functions returns 1 is 50% and the probability that it returns 2
    is 30%.

    Args:
        weights (1xN array): The vector with the probability of each index

        nonzero_ids (numpy array): Places where the weights vector is not
        zero.

    Returns:
        The randomly chosen index
    '''

    s = 0.0
    r = rand.random()
    for i in nonzero_ids:
        s += weights[i]
        if r <= s:
            return i

    raise RuntimeError('Failed to make a random choice. Check input vector.')


def meetFriend(A, s, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the random meeting model.

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    nonzero_ids = [np.nonzero(A[i, :])[0] for i in xrange(A.shape[0])]

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    # Cannot allow zero rows because rchoice() will fail
    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in trange(1, max_rounds):
        # Update the opinion for each node
        for i in range(N):
            r_i = rchoice(A[i, :], nonzero_ids[i])
            if r_i == i:
                op = s[i]
            else:
                op = z_prev[r_i]
            z[i] = (op + t*z_prev[i]) / (t+1)
        z_prev = z.copy()
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Meet a Friend converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'mf' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def meetFriend_nomem(A, s, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the random meeting model.

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A. This function does
    not save the opinions overtime and cannot generate a plot. However it uses
    very little memory and is useful for determining the final opinions and
    the convergence time of the model.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        t, z where t is the convergence time and z the vector of the
        final opinions.

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    nonzero_ids = [np.nonzero(A[i, :])[0] for i in xrange(A.shape[0])]

    z_prev = z.copy()

    if np.size(np.nonzero(A.sum(axis=1))) != N:
        raise ValueError("Matrix A has one or more zero rows")

    for t in trange(1, max_rounds):
        # Update the opinion for each node
        for i in range(N):
            r_i = rchoice(A[i, :], nonzero_ids[i])
            if r_i == i:
                op = s[i]
            else:
                op = z_prev[r_i]
            z[i] = (op + t*z_prev[i]) / (t+1)
        if conv_stop and \
           norm(z - z_prev, np.inf) < eps:
            print('Meet a Friend converged after {t} rounds'.format(t=t))
            break
        z_prev = z.copy()

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'mf' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, opinions=z)

    return t, z


def rand_matrices(A, t, nonzero_ids):
    '''Create random matrices A, B for the meetFriend_matrix model.

    The matrices are created as follows. A_t has a diagonal of (t-1)/t and B_t
    is the zero matrix. We make a weighted random choice for each node
    depending on the values of its row on A matrix. Depending on the outcome
    of this choice, either the B[i, i] = 1/t or A[i, r] = 1/t, where r is the
    random choice.

    Args:
        A (NxN numpy array): Weights matrix (its diagonal is the stubborness)

        t (int): Round number

    Returns:
        Two NxN matrices, A_t and B_t

    '''

    N = A.shape[0]
    A_t = sparse.lil_matrix((N, N))
    A_t.setdiag(np.ones(N) * (t-1)/t)
    B_t = sparse.lil_matrix((N, N))
    for i in xrange(N):
        r = rchoice(A[i, :], nonzero_ids[i])
        if r == i:
            B_t[i, i] = 1/t
        else:
            A_t[i, r] = 1/t

    return A_t.tocsr(), B_t.tocsr()


def meetFriend_matrix(A, max_rounds, norm_type=2, save=False):
    '''Simulates the random meeting model (matrix version).

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A. The function returns
    the distance from the equilibrium of the Friedkin-Johnsen over time.

    Args:
        A (NxN numpy array): Weights matrix (its diagonal is the stubborness)

        max_rounds (int): Maximum number of rounds to simulate

        norm_type: The norm type used to calculate the difference from the
        equilibrium

        save (bool): Save the simulation data into text files

    Returns:
        A vector containing the norm distances from the equlibrium of the
        Friedkin-Johnsen model.

    '''

    max_rounds = int(max_rounds)

    N = A.shape[0]
    B = np.diag(np.diag(A))

    nonzero_ids = [np.nonzero(A[i, :])[0] for i in xrange(A.shape[0])]

    equilibrium_matrix = np.dot(inv(np.eye(N) - (A - B)), B)
    R, Q = rand_matrices(A, 1, nonzero_ids)
    R = R + Q
    distances = np.zeros(max_rounds)
    for t in trange(2, max_rounds+2):
        A_t, B_t = rand_matrices(A, t, nonzero_ids)
        R = A_t.dot(R) + B_t
        distances[t-2] = norm(R - equilibrium_matrix, ord=norm_type)

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'mf' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=max_rounds, A=A, distances=distances,
                      norm=norm_type)

    return distances


def meetFriend_matrix_nomem(A, max_rounds, norm_type=2, save=False):
    '''Simulates the random meeting model (matrix version).

    Runs a maximum of max_rounds rounds of the "Meeting a Friend" model. If the
    model converges sooner, the function returns. The stubborness matrix of
    the model is extracted from the diagonal of matrix A. The function returns
    the distance from the equilibrium of the Friedkin-Johnsen when the process
    is complete. Uses less memory than the full model.

    Args:
        A (NxN numpy array): Weights matrix (its diagonal is the stubborness)

        max_rounds (int): Maximum number of rounds to simulate

        norm_type: The norm type used to calculate the difference from the
        equilibrium

        save (bool): Save the simulation data into text files

    Returns:
        The final distance from the equlibrium of the Friedkin-Johnsen model,
        using the specified norm.

    '''

    max_rounds = int(max_rounds)

    N = A.shape[0]
    B = np.diag(np.diag(A))

    nonzero_ids = [np.nonzero(A[i, :])[0] for i in xrange(A.shape[0])]

    equilibrium_matrix = np.dot(inv(np.eye(N) - (A - B)), B)
    R, Q = rand_matrices(A, 1, nonzero_ids)
    R = R + Q

    for t in trange(2, max_rounds+2):
        A_t, B_t = rand_matrices(A, t, nonzero_ids)
        R = A_t.dot(R) + B_t

    distance = norm(R - equilibrium_matrix, ord=norm_type)

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'mf' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=max_rounds, A=A, distance=distance,
                      norm=norm_type)

    return distance


def dynamic_weights(A, s, z, c, eps, p):
    '''Creates weighted edges based on the differences of opinions.

    The Generalized Asymmetric model works by using a dynamic weight matrix
    which is generated in each round. These weights are analogous to the
    proximity of the intrinsic belief of each node to the opinions of
    its neighbors.

    Args:
        A (NxN numpy array): Adjacency matrix (non-weighted)

        s (1xN numpy array): Intrinsic beliefs vector

        z (1xN numpy array): Current opinions vector

        c (string): Choose c function for the model. Possible choices are
        'simple', 'log', 'pow'.

        eps: Used in 'pow' func only

        p: Used in 'pow' func only

    Returns:
        The NxN matrix representing the weighted graph of the network.

    '''

    N = np.size(s)
    Q = np.zeros((N, N))

    functionDict = {
        'linear': lambda dist: 1 - dist,
        'log': lambda dist: 1 / np.log(dist + np.e),
        'pow': lambda dist: 1 / np.power(dist+eps, p)
    }

    cFunc = functionDict[c]

    for i in range(N):
        dist = np.abs(z - s[i])
        cResult = cFunc(dist)
        q = np.zeros(N)
        neighbors_i = A[i, :] > 0
        for node in np.flatnonzero(A[i, neighbors_i]):
            q[node] = cResult[node]/np.sum(cResult[neighbors_i])
        Q[i, :] = q

    return Q


def ga(A, B, s, max_rounds, eps=1e-6, conv_stop=True, save=False, **kwargs):
    '''Simulates the Generalized Asymmetric Coevolutionary Game.

    This model does nto require an adjacency matrix. Connections between
    nodes are calculated depending on the proximity of their opinions.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        B (NxN numpy array): The stubborness of each node

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

        **kargs: Arguments c, eps, and p for dynamic_weights function (eps and
        p need to be specified only if c='pow') (default: c='linear')

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    # Check if c function was specified
    if kwargs:
        c = kwargs['c']
        # Extra parameters for pow function
        eps_c = kwargs.get('eps', 0.1)
        p_c = kwargs.get('eps', 2)
    else:
        # Otherwise use linear as default
        c = 'linear'
        eps_c = None
        p_c = None

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # The matrix contains 0/1 values
    A_model = A.astype(np.int8)

    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        Q = dynamic_weights(A_model, s, z, c, eps_c, p_c) + B
        Q = row_stochastic(Q)
        B_temp = np.diag(np.diag(Q))
        Q = Q - B_temp
        z = Q.dot(z) + B_temp.dot(s)
        opinions[t, :] = z
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('G-A converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'ga' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, B=B, c=c, eps_c=eps_c,
                      p_c=p_c, opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def hk(s, op_eps, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the model of Hegselmann-Krause.

    This model does nto require an adjacency matrix. Connections between
    nodes are calculated depending on the proximity of their opinions.

    Args:
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        for i in range(N):
            # The node chooses only those with a close enough opinion
            friends_i = np.abs(z_prev - z_prev[i]) <= op_eps
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause converged after {t} rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'hk' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                  rounds_run=t+1, s=s, op_eps=op_eps,
                  opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def hk_perturbation(s, op_eps, max_rounds, eps=1e-6, conv_stop=True,
                    p_points=1, p_max=10):
    '''Simulates the model of Hegselmann-Krause (with stability test).

    This model does nto require an adjacency matrix. Connections between
    nodes are calculated depending on the proximity of their opinions. In this
    variation of the model we randomly choose some 'special' rounds during
    which the nodes can take into account opinions further than eps from their
    own.

    Args:
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        p_points (int): Number of points where the nodes will see further
        than op_eps

        p_max (int): Maximum round than can be a perturbation point

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s
    special_rounds = stdrand.sample(xrange(1, p_max), p_points)

    for t in trange(1, max_rounds):
        round_eps = 2*op_eps if t in special_rounds else op_eps
        for i in range(N):
            # The node chooses only those with a close enough opinion
            friends_i = np.abs(z_prev - z_prev[i]) <= round_eps
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause (perturbation) converged after {t} '
                  'rounds'.format(t=t))
            break

    # Graphically, the choice seems to 'happen' at the previous round
    # useful for plots
    special_rounds = [r-1 for r in special_rounds]

    return opinions[0:t+1, :], special_rounds


def hk_rand(s, K, op_eps, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulate the model of Hegselmann-Krause with random sampling.

    In each round every node chooses K other nodes uniformly at random and
    updates his opinion to be the average of the opinions of those K nodes
    that have a opinion distance at most equal to op_eps.

    Args:
        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        K (int): The number of nodes which will be randomly chosen in each
        round.

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''
    N, z, max_rounds = preprocessArgs(s, max_rounds)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        for i in range(N):
            # Choose K random nodes as temporary "neighbors"
            rand_sample = np.array(stdrand.sample(xrange(N), K))
            neighbors_i = np.zeros(N, dtype=bool)
            neighbors_i[rand_sample] = 1
            # Always choose yourself
            neighbors_i[i] = 1
            # The node chooses only those with a close enough opinion
            friends_i = np.abs(z_prev - z_prev[i]) <= op_eps
            friends_i = np.logical_and(neighbors_i, friends_i)
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause (random) converged after {t}'
                  ' rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'hk' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                  rounds_run=t+1, s=s, op_eps=op_eps,
                  opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def hk_local(A, s, op_eps, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the model of Hegselmann-Krause with an Adjacency Matrix.

    Contrary to the standard Hegselmann-Krause Model, here we make use of
    an adjacency matrix that represents an underlying social structure
    independent of the opinions held by the members of the society.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        for i in range(N):
            # Neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            opinion_close = np.abs(z_prev - z_prev[i]) <= op_eps
            # The node listens to those who share a connection with him
            # in the underlying network and also have an opinion
            # which is close to his own
            friends_i = np.logical_and(neighbor_i, opinion_close)
            z[i] = np.mean(z_prev[friends_i])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('Hegselmann-Krause (Local Knowledge) converged after {t} '
                  'rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'hkloc' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                  rounds_run=t+1, A=A, s=s, op_eps=op_eps,
                  opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def hk_local_nomem(A, s, op_eps, max_rounds, eps=1e-6, conv_stop=True,
                   save=False):
    '''Simulates the model of Hegselmann-Krause with an Adjacency Matrix

    Contrary to the standard Hegselmann-Krause Model, here we make use of
    an adjacency matrix that represents an underlying social structure
    independent of the opinions held by the members of the society. This
    variant does not store the intermediate opinions and as a result uses
    much less memory.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        op_eps: ε parameter of the model

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        t, z where t is the convergence time and z the vector of the
        final opinions.

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    z_prev = z.copy()

    for t in trange(1, max_rounds):
        for i in range(N):
            # Neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            opinion_close = np.abs(z_prev - z_prev[i]) <= op_eps
            # The node listens to those who share a connection with him
            # in the underlying network and also have an opinion
            # which is close to his own
            friends_i = np.logical_and(neighbor_i, opinion_close)
            z[i] = np.mean(z_prev[friends_i])
        if conv_stop and \
           norm(z - z_prev, np.inf) < eps:
            print('Hegselmann-Krause (Local Knowledge) converged after {t} '
                  'rounds'.format(t=t))
            break
        z_prev = z.copy()

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'hkloc' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                  rounds_run=t+1, A=A, s=s, op_eps=op_eps, opinions=z)

    return t, z


def kNN_static(A, s, K, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the static K-Nearest Neighbors Model.

    In this model, each node chooses his K-Nearest Neighbors during the
    averaging of his opinion.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        K (int): The number of the nearest neighbors to listen to

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        plot (bool): Plot preference (default: False)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        for i in range(N):
            # Find neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            # Sort the nodes by opinion distance
            sorted_dist = np.argsort(abs(z_prev - z_prev[i]))
            # Change the order of the logican neighbor_i array
            neighbor_i = neighbor_i[sorted_dist]
            # Keep only sorted neighbors
            friends_i = sorted_dist[neighbor_i]
            # In case that we have less than K friends numpy
            # will return the whole array (< K elements)
            k_nearest = friends_i[0:K]
            z[i] = np.mean(z_prev[k_nearest])
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('K-Nearest Neighbors (static) converged after {t} '
                  'rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'kNNs' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, K=K,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def kNN_static_nomem(A, s, K, max_rounds, eps=1e-6, conv_stop=True,
                     save=False):
    '''Simulates the static K-Nearest Neighbors Model. Reduced memory usage.

    In this model, each nodes chooses his K-Nearest Neighbors during the
    averaging of his opinion. This variant does not store the intermediate
    opinions and as a result uses much less memory.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        K (int): The number of the nearest neighbors to listen to

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        t, z where t is the convergence time and z the vector of the
        final opinions.

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()

    for t in trange(1, max_rounds):
        Q = np.zeros((N, N))
        for i in xrange(N):
            # Find neighbors in the underlying social network
            neighbor_i = A_model[i, :] > 0
            # Sort the nodes by opinion distance
            sorted_dist = np.argsort(abs(z_prev - z_prev[i]))
            # Change the order of the logical neighbor_i array
            neighbor_i = neighbor_i[sorted_dist]
            # Keep only sorted neighbors
            friends_i = sorted_dist[neighbor_i]
            # In case that we have less than K friends numpy
            # will return the whole array (< K elements)
            k_nearest = friends_i[0:K]
            Q[i, k_nearest] = 1/k_nearest.size
        z = Q.dot(z_prev)
        if conv_stop and \
           norm(z - z_prev, np.inf) < eps:
            print('K-Nearest Neighbors (static) converged after {t} '
                  'rounds'.format(t=t))
            break
        z_prev = z.copy()

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'kNNs' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, K=K, opinions=z, Q=Q)

    return t, z, Q


def kNN_dynamic(A, s, K, max_rounds, eps=1e-6, conv_stop=True, save=False):
    '''Simulates the dynamic K-Nearest Neighbors Model.

    In this model, each nodes chooses his K-Nearest Neighbors during the
    averaging of his opinion. The adjacency matrix changes between rounds
    depending on the opinions.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        K (int): The number of the nearest neighbors to listen to

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        A txN vector of the opinions of the nodes over time

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()
    opinions = np.zeros((max_rounds, N))
    opinions[0, :] = s

    for t in trange(1, max_rounds):
        Q = np.zeros((N, N))
        # TODO: Verify that this contains the original paths of A
        A_squared = A_model.dot(A_model)
        for i in range(N):
            # Find 2-neighbors in the underlying social network
            neighbor2_i = A_squared[i, :] > 0
            # Sort the nodes by opinion distance
            sorted_dist = np.argsort(abs(z_prev - z_prev[i]))
            # Change the order of the logican neighbor2_i array
            neighbor2_i = neighbor2_i[sorted_dist]
            # Keep only sorted neighbors
            friends_i = sorted_dist[neighbor2_i]
            # In case that we have less than K friends numpy
            # will return the whole array (< K elements)
            k_nearest = friends_i[0:K]
            Q[i, k_nearest] = 1/k_nearest.size
            z[i] = np.mean(z_prev[k_nearest])
        A_model = Q.copy()
        opinions[t, :] = z
        z_prev = z.copy()
        if conv_stop and \
           norm(opinions[t - 1, :] - opinions[t, :], np.inf) < eps:
            print('K-Nearest Neighbors (dynamic) converged after {t} '
                  'rounds'.format(t=t))
            break

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'kNNd' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, K=K,
                      opinions=opinions[0:t+1, :])

    return opinions[0:t+1, :]


def kNN_dynamic_nomem(A, s, K, max_rounds, eps=1e-6, conv_stop=True,
                      save=False):
    '''Simulates the dynamic K-Nearest Neighbors Model. Reduced Memory.

    In this model, each nodes chooses his K-Nearest Neighbors during the
    averaging of his opinion. Opinions over time are not saved.

    Args:
        A (NxN numpy array): Adjacency matrix (its diagonal is the stubborness)

        s (1xN numpy array): Initial opinions (intrinsic beliefs) vector

        K (int): The number of the nearest neighbors to listen to

        max_rounds (int): Maximum number of rounds to simulate

        eps (double): Maximum difference between rounds before we assume that
        the model has converged (default: 1e-6)

        conv_stop (bool): Stop the simulation if the model has converged
        (default: True)

        save (bool): Save the simulation data into text files

    Returns:
        t, z, Q where t is the convergence time, z the vector of the
        final opinions and Q the final adjacency matrix of the network

    '''

    N, z, max_rounds = preprocessArgs(s, max_rounds)

    # All nodes must listen to themselves for the averaging to work
    A_model = A + np.eye(N)

    # The matrix contains 0/1 values
    A_model = A_model.astype(np.int8)

    z_prev = z.copy()

    for t in trange(1, max_rounds):
        Q = np.zeros((N, N))
        # TODO: Verify that this contains the original paths of A
        A_squared = A_model.dot(A_model)
        for i in range(N):
            # Find 2-neighbors in the underlying social network
            neighbor2_i = A_squared[i, :] > 0
            # Sort the nodes by opinion distance
            sorted_dist = np.argsort(abs(z_prev - z_prev[i]))
            # Change the order of the logican neighbor2_i array
            neighbor2_i = neighbor2_i[sorted_dist]
            # Keep only sorted neighbors
            friends_i = sorted_dist[neighbor2_i]
            # In case that we have less than K friends numpy
            # will return the whole array (< K elements)
            k_nearest = friends_i[0:K]
            Q[i, k_nearest] = 1/k_nearest.size
            z[i] = np.mean(z_prev[k_nearest])
        A_model = Q.copy()
        if conv_stop and \
           norm(z - z_prev, np.inf) < eps:
            print('K-Nearest Neighbors (dynamic) converged after {t} '
                  'rounds'.format(t=t))
            break
        z_prev = z.copy()

    if save:
        timeStr = datetime.now().strftime("%m%d%H%M")
        simid = 'kNNd' + timeStr
        save_data(simid, N=N, max_rounds=max_rounds, eps=eps,
                      rounds_run=t+1, A=A, s=s, K=K, opinions=z, Q=Q)

    return t, z, Q
