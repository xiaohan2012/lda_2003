from __future__ import division

import numpy as np

from scipy.special import (psi, polygamma)

from util import close_enough


def init_ips(M, K, alpha, docs):
    N = np.asarray([D_n.size for D_n in docs],
                   dtype=np.float64)
    return (np.ones((M, K), dtype=np.float64) * alpha.reshape(1, K)
            + N.reshape(M, 1) / K)


def e_step_one_iter(alpha, beta, docs, phi, ips):
    M, K = docs.size, alpha.size                                        

    for m in xrange(M):
        N_m = docs[m].size
        psi_sum_ips = psi(ips[m, :].sum())
        for n in xrange(N_m):
            for i in xrange(K):
                E_q = psi(ips[m, i]) - psi_sum_ips
                phi[m][n, i] = (beta[i, docs[m][n]] *
                                np.exp(E_q))
        phi[m] /= phi[m].sum(axis=1)[:, None]  # normalize phi
        ips[m, :] = alpha + phi[m].sum(axis=0)

    # gradient computation
    grad_ips = np.zeros(ips.shape, dtype=np.float64)
    for m in xrange(M):
        for i in xrange(K):
            # print("{}+ {} - {} = {}".format(alpha[i], phi[m][:, i].sum(), ips[m, i],
            #                                 alpha[i] + phi[m][:, i].sum() - ips[m, i]))
            grad_ips[m, i]\
                = (polygamma(1, ips[m, i]) * (alpha[i] + phi[m][:, i].sum() - ips[m, i]) -
                   polygamma(1, ips[m, :].sum()) * (alpha.sum() + phi[m].sum() - ips[m, :].sum()))
            print(grad_ips[m, i])

    return (phi, ips, grad_ips)


def e_step(alpha, beta, docs):
    """
    E step in the LDA variational inference for training

    Parameter:
    ------------

    Return:
    ------------
    ips: (M, K)
    phi: (M, N_m, K)
    """
    M, K = docs.size, alpha.size
    ips = init_ips(M, K, alpha, docs)

    # phi to be returned
    phi = np.array([np.zeros((docs[m].size, K), dtype=np.float64)
                    for m in xrange(M)],
                   dtype=np.object)

    while True:
        phi, ips, grad_ips = e_step_one_iter(alpha, beta,
                                             docs, phi, ips)

        # check for convergence
        if np.abs(grad_ips).max() <= 1e-5:
            break
    
    return ips, phi


def update_beta(K, M, V, docs, phi):
    # TODO: add smoothing 
    beta = np.zeros((K, V), dtype=np.float64)
    
    # updating beta
    for i in xrange(K):
        for m in xrange(M):
            for n in xrange(docs[m].size):
                j = docs[m][n]
                beta[i, j] += phi[m][n, i]
    return beta / beta.sum(axis=1)[:, None]


def gradient_g(M, alpha, ips):
    """
    Return:
    -----------
    numpy.matrix: 2d
    
    """
    psi_sum_alpha = psi(alpha.sum())
    sum_psi_ips = psi(ips.sum(axis=1)).sum()
    return (M * (psi_sum_alpha - psi(alpha)) +
            psi(ips).sum(axis=0) -
            sum_psi_ips)


def hessian_h_and_z(M, alpha):
    h = M * polygamma(1, alpha)
    z = - polygamma(1, alpha.sum())
    return h, z


def update_alpha(M, ips, alpha):
    # Unsolved
    old_alpha = alpha

    max_iter = 10
    iter = 0
    while True:
        iter += 1
        if iter >= max_iter:
            break

        print old_alpha
        g = gradient_g(M, old_alpha, ips)
        h, z = hessian_h_and_z(M, old_alpha)

        print 'g:', g
        print 'h:', h
        print

        c = (g / h).sum() / (1./z + (1./h).sum())
        
        new_alpha = old_alpha - (g - c) / h

        if close_enough(old_alpha, new_alpha):
            break
        else:
            old_alpha = new_alpha

    return new_alpha
    

def m_step(ips, phi, docs, V):
    """
    Parameter:
    -------------

    Return:
    --------------
    alpha, beta

    """
    M, K = ips.shape
    beta = update_beta(K, M, docs, phi)
    alpha = update_alpha(K, ips)
    return alpha, beta
    

def train(docs, alpha, beta, K, V):
    """
    Parameter:
    --------------
    
    docs: numpy.ndarray
        list of documents, each document
        represented by its sequence of token ids
    alpha: length K array
        Dirichlet parameter, intial value is passed
    beta: K x V matrix
        topc midels, intial value is passed
    """
    M, V = docs.shape
    # variational Dirichet parameter
    # which generates theta
    # M x K matrix
    ips = np.zeros((M, K), dtype=np.float64)
    
    # variational multinomial parameter
    # which generates z
    # M x K matrix
    phi = np.zeros((M, K), dtype=np.float64)

    converge = False
    while converge:
        # E step
        # maximize the lower bound
        # in terms of ips and phi
        ips, phi = e_step(alpha, beta, docs)
        
        # M step
        # maximize the lower bound
        # in terms of alpha and beta
        alpha, beta = m_step(ips, phi, docs)


if __name__ == "__main__":
    M = 3
    K = 2
    V = 6
    alpha = np.asarray([1, 2], dtype=np.float64)
    docs = np.array([
        np.arange(2, 4),  # 2~3
        np.arange(3, 6),   # 3~5
        np.arange(0, 4),  # 0~3
    ], dtype=np.object)
    beta = np.array([[.1, .2, .1, .1, .2, .3],
                     [.3, .1, .2, .2, .1, .1]])
    beta = np.random.random((K, V))
    beta /= beta.sum(axis=1)[:, None]
    print(beta)
    ips = init_ips(M, K, alpha, docs)
        
    phi = np.array([np.zeros((docs[m].size, K), dtype=np.float64)
                    for m in xrange(M)],
                   dtype=np.object)
    
    for i in xrange(1):
        phi, ips, grad_ips = e_step_one_iter(alpha, beta,
                                             docs, phi, ips)
        print grad_ips
