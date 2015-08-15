import numpy as np
from lda_2003.util import close_enough
from nose.tools import (assert_true, assert_false)


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
