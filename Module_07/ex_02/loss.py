""" Understand and manipulate loss function for multivariate linear regression. """
import numpy as np


def _args_are_valid_arrays(function):
    """ Little generator for error handling """
    def wrapper(*args, **kwargs):
        """ Wrapper function to make sure the parameters are 
            numpy ndarrays of valid dimensions and type

        Args:
            *args (numpy.ndarray): vectors of dimension m * 1

        Returns:
            bool: function if args are of the desired type and dimensions, None otherwise
        """
        for arg in args:
            if not isinstance(arg, np.ndarray):
                return None
            if not (np.issubdtype(arg.dtype, np.integer) or np.issubdtype(arg.dtype, np.floating)):
                return None
            if arg.size == 0:
                return None
        return function(*args, **kwargs)
    return wrapper


@_args_are_valid_arrays
def loss_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
    """ Computes the half mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.

    Args:
        y (numpy.ndarray): vector of shape (m, 1), the real values
        y_hat: has to be an numpy.array, a vector, the predicted values

    Return:
        mse (float): float, the mean squared error of the two vectors.
        None if y or y_hat are not of the required dimensions or type.
    """
    if y.shape != y_hat.shape or y_hat.shape not in [(y_hat.size, 1), (y_hat.size, )]:
        return None
    loss = np.sum(np.square(y_hat - y)) / (len(y_hat) * 2)
    return loss


def tests():
    """ Little test function """
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    result = loss_(X, Y)
    expected = 2.142857142857143
    np.testing.assert_equal(result, expected)

    # Example 2:
    result = loss_(X, X)
    expected = 0.0
    np.testing.assert_equal(result, expected)

    # Example 3
    X = np.array([[34], [37], [44], [47], [48], [48],
                 [46], [43], [32], [27], [26], [24]])
    Y = np.array([[37], [40], [46], [44], [46], [50],
                 [45], [44], [34], [30], [22], [23]])
    result = loss_(X, Y)
    expected = 2.9583333333333335
    np.testing.assert_equal(result, expected)

    print("All tests passing, no assert raised, gg")


if __name__ == "__main__":
    tests()
