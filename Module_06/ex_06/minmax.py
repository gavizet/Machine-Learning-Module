""" Introduction to standardization/normalization methods with min-max """
import numpy as np


def _is_valid_ndarray_vector(x: np.ndarray) -> bool:
    """ Make sure x is a numpy ndarrays of valid dimensions and type

    Args:
        x (numpy.ndarray): vector of dimension m * 1

    Returns:
        bool: True if x is valid, False otherwise
    """
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return False
    if not np.issubdtype(x.dtype, np.floating) and not np.issubdtype(x.dtype, np.integer):
        return False
    if x.shape not in [(x.size, ), (x.size, 1)]:
        return False
    return True


def minmax(x: np.ndarray) -> np.ndarray | None:
    """Computes the normalized version of a vector x using the min-max standardization.

    Args:
        x (numpy.ndarray): vector of values.

    Returns:
        x_prime (numpy.ndarray): vector of normalized values.
        None if x is not a numpy.ndarray, is empty, or is not of vector dimensions.
    """
    if not _is_valid_ndarray_vector(x):
        return None


def main_tests():
    # Example 1:
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    result1 = minmax(X)
    expected1 = np.array([[0.58333333,
                           1.,
                           0.33333333,
                           0.77777778,
                           0.91666667,
                           0.66666667,
                           0.]])
    np.testing.assert_array_almost_equal(result1, expected1)

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    result2 = minmax(Y)
    expected2 = np.array([0.63636364,
                          1.,
                          0.18181818,
                          0.72727273,
                          0.93939394,
                          0.6969697,
                          0.])
    np.testing.assert_array_almost_equal(result2, expected2)


if __name__ == "__main__":
    main_tests()
