import numpy as np

from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_true, assert_equal)

from scipy.special import (psi, polygamma)

from lda_2003.lda import (init_ips,
                          gradient_g,
                          hessian_h_and_z,
                          update_alpha,
                          update_beta,
                          e_step,
                          e_step_one_iter,
                          lower_bound,
                          m_step,
                          train)


M = 3
K = 2
alpha = np.asarray([1, 2], dtype=np.float64)
docs = np.array([
    np.arange(2, 4),  # 2~3
    np.arange(3, 6),  # 3~5
    np.arange(0, 4),  # 0~3
], dtype=np.object)
V = 6
phi = np.array([
    np.array([[.1, .9],
              [.4, .6]]),  # 2~3
    np.array([[.3, .7],
              [.7, .3],
              [.5, .5]]),  # 3~5
    np.array([[.4, .6],
              [.5, .5],
              [.7, .3],
              [.9, .1]]),  # 0~3
], dtype=np.object)
beta = np.array([[.1, .2, .1, .1, .2, .3],
                 [.3, .1, .2, .2, .1, .1]])


def test_init_ips():
    actual_ips = init_ips(M, K, alpha, docs)

    expected_ips = np.asarray([[1 + 2. / K, 2 + 2. / K],
                               [1 + 3. / K, 2 + 3. / K],
                               [1 + 4. / K, 2 + 4. / K]],
                              dtype=np.float64)
    assert_array_almost_equal(actual_ips, expected_ips)


def test_gradiant_g():
    ips = init_ips(M, K, alpha, docs)
    g = gradient_g(M, alpha, ips)

    for i in xrange(K):
        expected = (M * (psi(alpha.sum()) - psi(alpha[i])) +
                    np.sum([(psi(ips[d, i]) -
                             psi(ips[d, :].sum()))
                            for d in xrange(M)]))
        actual = g[i]
        assert_array_almost_equal(expected, actual)


def test_hessian_h_and_z():
    h, z = hessian_h_and_z(M, alpha)
    for i in xrange(alpha.size):
        actual = h[i]
        expected = - M * polygamma(1, alpha[i])
        assert_array_almost_equal(actual, expected)
    assert_array_almost_equal(z, M * polygamma(1, alpha.sum()))


def test_update_alpha():
    ips = init_ips(M, K, alpha, docs)
    
    new_alpha = update_alpha(M, ips, alpha)

    # gradient should be close to zeros after convergence
    assert_array_almost_equal(gradient_g(M, new_alpha, ips),
                              np.zeros(new_alpha.shape))


def test_update_beta():
    actual = update_beta(K, M, V, docs, phi)
    
    # calculated by hand

    assert_array_almost_equal(actual[:, 2],
                              np.array([0.8, 1.2]) / np.array([4.5, 4.5]))


def test_e_step_one_iter():
    ips = init_ips(M, K, alpha, docs)
    phi = np.array([np.zeros((docs[m].size, K), dtype=np.float64)
                    for m in xrange(M)],
                   dtype=np.object)

    # don't know why it converges just afer 1 iteration
    for i in xrange(1):
        phi, ips, grad_ips = e_step_one_iter(alpha, beta,
                                             docs, phi, ips)
        
    assert_true(np.abs(grad_ips).max() <= 1e-5)
        

def test_e_step():
    old_ips = init_ips(M, K, alpha, docs)
    new_ips, phi = e_step(alpha, beta, docs)

    # very permissive test(not sure how to test it better):
    # make sure new_ips changes
    assert_true(np.abs(new_ips - old_ips).min() >= 1e-5)

    # make sure sum to one
    for m in xrange(phi.size):
        assert_array_almost_equal(phi[m].sum(axis=1), 1)

    # make sure the lower bound increases
    old_lb_val = lower_bound(old_ips, phi, alpha, beta, docs, V)
    new_ips, new_phi = e_step(alpha, beta, docs)
    new_lb_val = lower_bound(new_ips, new_phi, alpha, beta, docs, V)

    assert_true(new_lb_val >= old_lb_val)


def test_m_step():
    # make sure that the lower bound increases
    ips = init_ips(M, K, alpha, docs)
    alpha_ = alpha
    old_lb_val = lower_bound(ips, phi, alpha, beta, docs, V)

    alpha_, beta_ = m_step(ips, phi, alpha, docs, V)
    new_lb_val = lower_bound(ips, phi, alpha_, beta_, docs, V)
    
    assert_true(new_lb_val >= old_lb_val)


def test_lower_bound():
    ips = init_ips(M, K, alpha, docs)
    actual = lower_bound(ips, phi, alpha, beta, docs, V)

    # how to test if we don't want to calculate by hand
    # 1st, <= 0
    assert_true(actual <= 0)

    # 2nd, after training several iterations
    # the value should be increasing
    # as our goal is to maximize the lower bound
    old_lb_val = actual
    alpha_, beta_ = alpha, beta
    ips_, phi_ = ips, phi
    for i in xrange(10):
        ips_, phi_ = e_step(alpha_, beta_, docs)
        alpha_, beta_ = m_step(ips_, phi_, alpha_, docs, V)

        new_lb_val = lower_bound(ips_, phi_, alpha_, beta_, docs, V)
        assert_true(new_lb_val >= old_lb_val)

        old_lb_val = new_lb_val


def test_train():
    alpha_, beta_, ips_, phi_, lower_bound_values\
        = train(docs, alpha, beta, K, V, max_iter=50)

    # some dimensionality checking
    assert_equal(alpha_.shape, (K, ))
    assert_equal(beta_.shape, (K, V))
    assert_equal(ips_.shape, (M, K))

    for m in xrange(M):
        assert_equal(phi_[m].shape, (docs[m].size, K))

    for prev, cur in zip(lower_bound_values, lower_bound_values[1:]):
        assert_true(prev <= cur)
