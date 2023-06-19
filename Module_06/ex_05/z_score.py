""" Introduction to standardization/normalization methods with z-score """
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


def zscore(x: np.ndarray) -> np.ndarray | None:
    """Computes the normalized version of a vector x using the z-score standardization.

    Args:
        x (numpy.ndarray): vector of values.

    Returns:
        x_prime (numpy.ndarray): vector of normalized values.
        None if x is not a numpy.ndarray, is empty, or is not of vector dimensions.
    """
    if not _is_valid_ndarray_vector(x):
        return None
    mean = x.mean()
    std = x.std()
    x_prime = (x - mean) / std
    return x_prime


def main_tests():
    # Example 1:
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    result1 = zscore(X)
    expected1 = np.array([[-0.08620324],
                         [1.2068453],
                         [-0.86203236],
                         [0.51721942],
                         [0.94823559],
                         [0.17240647],
                         [-1.89647119]])
    np.testing.assert_array_almost_equal(result1, expected1)

    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    result2 = zscore(Y)
    expected2 = np.array([0.11267619,
                          1.16432067,
                          -1.20187941,
                          0.37558731,
                          0.98904659,
                          0.28795027,
                          -1.72770165])
    np.testing.assert_array_almost_equal(result2, expected2)
    print("Tests for z_score.py raised no assert error, gg.")


if __name__ == "__main__":
    main_tests()
