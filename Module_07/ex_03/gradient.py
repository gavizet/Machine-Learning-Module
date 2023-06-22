""" Understand and manipulate concept of gradient in the case of multivariate formulation. """
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
def gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
    """ Computes a gradient vector from three non-empty numpy.array.

    Args:
        x (numpy.ndarray): matrix m * n, 
        y (numpy.ndarray): vector m * 1.
        theta (numpy.ndarray): vector (n + 1) * 1.

    Return:
        gradient (numpy.ndarray): vector of dimensions n * 1, 
        containg the result of the formula for all J (loss).
        None if y or y_hat are not of the required dimensions or type.
    """
    pass


def tests():
    """ Little test function """
    x = np.array([[-6, -7, -9],
                  [13, -2, 14],
                  [-7, 14, -1],
                  [-8, -4, 6],
                  [-5, -9, 6],
                  [1, -5, 11],
                  [9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([3, 0.5, -6]).reshape((-1, 1))

    # Example 1:
    gradient(x, y, theta1)
    # expected = np.array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]])

    # Example 2:
    theta2 = np.array([0, 0, 0]).reshape((-1, 1))
    gradient(x, y, theta2)
    # expected = np.array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])


if __name__ == "__main__":
    tests()
