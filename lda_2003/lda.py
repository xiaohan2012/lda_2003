from __future__ import division

import numpy as np

from scipy.special import (psi, polygamma, gamma, gammaln)

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
        ips[m] = alpha + phi[m].sum(axis=0)


    # gradient computation
    grad_ips = np.zeros(ips.shape, dtype=np.float64)
    for m in xrange(M):
        for i in xrange(K):
            grad_ips[m, i]\
                = (polygamma(1, ips[m, i]) * (alpha[i] + phi[m][:, i].sum() - ips[m, i]) -
                   polygamma(1, ips[m, :].sum()) * (alpha.sum() + phi[m].sum() - ips[m, :].sum()))

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
    h = - M * polygamma(1, alpha)
    z = M * polygamma(1, alpha.sum())
    return h, z


def update_alpha(M, ips, alpha):
    old_alpha = alpha

    max_iter = 10
    iter = 0
    while True:
        iter += 1
        if iter >= max_iter:
            break

        # print old_alpha
        g = gradient_g(M, old_alpha, ips)
        h, z = hessian_h_and_z(M, old_alpha)

        # print 'g:', g
        # print 'h:', h
        # print

        c = (g / h).sum() / (1./z + (1./h).sum())
        
        new_alpha = old_alpha - (g - c) / h

        if close_enough(old_alpha, new_alpha):
            break
        else:
            old_alpha = new_alpha

    return new_alpha
    

def m_step(ips, phi, alpha, docs, V):
    """
    Parameter:
    -------------

    Return:
    --------------
    alpha, beta

    """
    M, K = ips.shape
    beta = update_beta(K, M, V, docs, phi)
    alpha = update_alpha(M, ips, alpha)
    return alpha, beta
    

def lower_bound(ips, phi, alpha, beta, docs, V):
    K = ips.shape[1]
    M = docs.size

    ret = 0

    # it will be reused later
    psi_ips_2d = (psi(ips) - psi(ips.sum(axis=1))[:, None])
        
    # 1st line
    ret += M * (gammaln(alpha.sum()) - gammaln(alpha).sum())
    ret += np.sum((alpha - 1) * psi_ips_2d)
    
    assert not np.isinf(ret) and not np.isnan(ret), '1st line'

    # 2nd line
    for m in xrange(M):
        ret += np.sum(
            (np.asmatrix(phi[m]) *
             np.asmatrix(psi_ips_2d[m, :].reshape(K, 1)))
        )
    
    assert not np.isinf(ret) and not np.isnan(ret), '2nd line'

    # 3rd line
    for m in xrange(M):
        for n in xrange(docs[m].size):
            for i in xrange(K):
                ret += phi[m][n, i] * np.log(beta[i, docs[m][n]])

    assert not np.isinf(ret) and not np.isnan(ret), '3rd line'

    # 4th line
    # 1st term
    tmp = np.sum(gammaln(ips.sum(axis=1)))

    ret -= np.sum(gammaln(ips.sum(axis=1)))

    assert not np.isinf(ret) and not np.isnan(ret), '4th line, 1st term'

    # 2nd term
    ret += gammaln(ips).sum()
    
    assert not np.isinf(ret) and not np.isnan(ret), '4th line, 2nd term'

    # 3rd term
    ret -= np.sum((ips - 1) * psi_ips_2d)

    assert not np.isinf(ret) and not np.isnan(ret), '4th line, 3rd term'

    # 5th line
    for m in xrange(M):
        ret -= np.sum(phi[m] * np.log(phi[m]))
    
    assert not np.isinf(ret) and not np.isnan(ret), '5th line'

    return ret


def train(docs, alpha, beta, K, V, max_iter):
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
    M = docs.size
    # variational Dirichet parameter
    # which generates theta
    # M x K matrix
    ips = np.zeros((M, K), dtype=np.float64)

    # variational Multinomial parameter
    # for each document and each word in the document
    # will be set during the training
    # shape: (M, N_m, K)
    phi = None

    # this ensures the while test passes at the first time
    old_lower_bound_value = 1000
    new_lower_bound_value = 0
    
    lower_bound_values = []

    i = 0
    while True:
        # E step
        # maximize the lower bound
        # in terms of ips and phi
        ips, phi = e_step(alpha, beta, docs)

        # M step
        # maximize the lower bound
        # in terms of alpha and beta
        alpha, beta = m_step(ips, phi, alpha, docs, V)
                
        if (np.abs(new_lower_bound_value - old_lower_bound_value) <= 1e-3):
            break

        old_lower_bound_value = new_lower_bound_value
        new_lower_bound_value = lower_bound(ips, phi, alpha, beta, docs, V)

        lower_bound_values.append(new_lower_bound_value)
        
        i += 1
        print "At iter {:3d}, lower bound {}".format(i, new_lower_bound_value)
        if i == max_iter:
            break

    return (alpha, beta, ips, phi, lower_bound_values)


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

    ips = init_ips(M, K, alpha, docs)
    update_alpha(M, ips, alpha)

    # print(beta)
    # ips = init_ips(M, K, alpha, docs)
        
    # phi = np.array([np.zeros((docs[m].size, K), dtype=np.float64)
    #                 for m in xrange(M)],
    #                dtype=np.object)
    
    # for i in xrange(1):
    #     phi, ips, grad_ips = e_step_one_iter(alpha, beta,
    #                                          docs, phi, ips)
    #     print grad_ips
