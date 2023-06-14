""" Understand and manipulate the notion of hypothesis in machine learning.
You must implement a function which adds an extra column of '1' on the left side 
of a given vector or matrix """
import numpy as np

# Instead of np.c_[], we could use one of the np.stack methods :
# (stack, hstack, vstack, dstack, column_stack),
# which are all helper functions built on top of np.concatenate.
# See numpy doc and these stackoverflow posts for more details :
# https://stackoverflow.com/questions/53876020/how-to-efficiently-add-a-column-in-a-multidimensional-numpy-array-with-different
# https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array


def add_intercept(x: np.ndarray) -> np.ndarray or None:
    """Adds a column of '1' to the non-empty numpy.array x.
    Args:
        array: has to be a numpy.array of dimension m * n.

    Returns:
        X (numpy.array): of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.

    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0 or \
        (not np.issubdtype(x.dtype, np.floating) and
            not np.issubdtype(x.dtype, np.integer)):
        return None
    ones_col = np.ones((x.shape[0], 1))
    X = np.c_[ones_col, array]
    return X


def main():
    # Test 1
    x = np.arange(1, 6)
    np.testing.assert_array_equal(add_intercept(x), np.array([[1., 1.],
                                                              [1., 2.],
                                                              [1., 3.],
                                                              [1., 4.],
                                                              [1., 5.]]))

    # Test 2
    y = np.arange(1, 10).reshape((3, 3))
    np.testing.assert_array_equal(add_intercept(y), np.array([[1., 1., 2., 3.],
                                                              [1., 4., 5., 6.],
                                                              [1., 7., 8., 9.]]))


if __name__ == "__main__":
    main()

# Pytest test file can be found in tests/Module_05/test_ex03.py
