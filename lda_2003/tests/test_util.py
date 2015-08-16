import os
import numpy as np
from lda_2003.util import (close_enough, load_line_corpus)
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
                 [u'product', u'ab', u'defin',
                  u'number', u'column', u'figur',
                  u'right', u'illustr', u'diagrammat',
                  u'product', u'two', u'matric'])
