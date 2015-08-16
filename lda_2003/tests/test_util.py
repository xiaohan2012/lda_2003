import os
import numpy as np
from numpy.testing import assert_array_almost_equal

from lda_2003.util import (close_enough, load_line_corpus, vectorize_docs)

from nose.tools import (assert_true, assert_false, assert_equal)

CURDIR = os.path.dirname(os.path.realpath(__file__))


def test_close_enough_True():
    old_x = np.arange(10, dtype=np.float64)
    new_x = np.arange(10, dtype=np.float64)
    assert_true(close_enough(old_x, new_x))


def test_close_enough_False():
    reltol = 1e-5
    old_x = np.arange(10, dtype=np.float64)
    
    new_x = np.arange(10, dtype=np.float64)

    # first element a little higher than reltol
    new_x[0] += (1 + reltol) * reltol

    assert_false(close_enough(old_x, new_x, reltol))


def test_load_line_corpus():
    docs = load_line_corpus(CURDIR + '/data/corpus.txt')
    assert_equal(len(docs), 2)
    assert_equal(docs[0],
                 [u'product', u'defin',
                  u'number', u'column', u'figur',
                  u'right', u'illustr', u'diagrammat',
                  u'product', u'two', u'matric'])


def test_vectorize_docs():
    docs = load_line_corpus(CURDIR + '/data/corpus.txt')
    mat, vocab = vectorize_docs(docs)
    print vocab
    assert_equal(vocab[0], u'product')
    assert_equal(vocab[1], u'right')
    assert_equal(len(vocab), 14)

    assert_equal(len(mat), 2)

    for doc, doc_m in zip(docs, mat):
        assert_equal([vocab[w] for w in doc_m], doc)
        
