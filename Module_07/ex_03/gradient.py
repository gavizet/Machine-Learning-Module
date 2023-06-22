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
        x (numpy.ndarray): matrix m * n, training examples, m = num of values, n = num of features
        y (numpy.ndarray): vector m * 1, predicted values
        theta (numpy.ndarray): vector (n + 1) * 1, our parameters

    Return:
        gradient (numpy.ndarray): vector (n + 1) * 1
        containg the result of the formula for all J (loss).
        None if y or y_hat are not of the required dimensions or type.
    """
    m, n = x.shape
    # Check vectors are of valid dimension and compatible shape with matrix x
    if y.shape not in [(m, 1), (m, )] or theta.shape not in [(n, 1), (n, )]:
        print("Hey, should enter here")
        return None
    # Add column of 1s left of Matrix so we can vectorize the equation
    # (we just scale theta_0 by 1). x_prime is of dimension m * n + 1
    x_prime = np.c_[np.ones((x.shape[0], 1)), x]
    # Get predicted values, vector m * 1
    y_hat = x.dot(theta)
    # Transpose X' so it is now (n + 1) * m
    x_transpose = np.transpose(x_prime)
    # Do the dot product between X'T (n + 1, m) and vector (m, 1). Resulting shape is (n + 1, 1)
    gradient_vector = (x_transpose.dot(y_hat - y)) / (x_prime.shape[0])
    return gradient_vector


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
    result = gradient(x, y, theta1)
    expected = np.array(
        [[-33.71428571], [-37.35714286], [183.14285714], [-393.]])
    np.testing.assert_array_almost_equal(result, expected)

    # Example 2:
    theta2 = np.array([0, 0, 0]).reshape((-1, 1))
    result = gradient(x, y, theta2)
    expected = np.array([[-0.71428571], [0.85714286],
                        [23.28571429], [-26.42857143]])
    np.testing.assert_array_almost_equal(result, expected)

    # If result is None, i.e the dimensions / type of an argument is not valid,
    # the np.testing.assert will throw a TypeError as we are trying to compare
    # a NoneType (result of gradient()) and a float(expected numpy array result)
    # This is normal and expected behavior.
    print("All tests passing, no assert raised, gg")


if __name__ == "__main__":
    tests()
