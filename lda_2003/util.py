import numpy as np


def close_enough(old_x, new_x, abstol=10e-5):
    """
    Check if differentce between old_x and new_x is close enough.

    Being close enough is the same as
    the maximum difference betwen old_x and new_x is below abstol

    Parameter:
    ---------------

    old_x: numpy.ndarray
    
    new_x: numpy.ndarray
        same shape as new_x

    Return:
    -----------------
    bool

    """
    return np.abs((old_x - new_x)).max() <= abstol
    
    
