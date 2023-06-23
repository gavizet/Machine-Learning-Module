""" Understand and manipulate the concept of gradient descent in the case of 
multivariate linear regression """
import numpy as np
from Module_07.ex_01.prediction import predict_


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
        x (numpy.ndarray): matrix m * n + 1, training examples, 
                            m = num of values, n = num of features
        y (numpy.ndarray): vector m * 1, predicted values
        theta (numpy.ndarray): vector (n + 1) * 1, our parameters

    Return:
        gradient (numpy.ndarray): vector (n + 1) * 1
        containg the result of the formula for all J (loss).
        None if y or y_hat are not of the required dimensions or type.
    """
    # Get predicted values, vector m * 1
    y_hat = x.dot(theta)
    # Transpose X' so it is now (n + 1) * m
    x_transpose = np.transpose(x)
    # Do the dot product between X'T (n + 1, m) and vector (m, 1). Resulting shape is (n + 1, 1)
    gradient_vector = (x_transpose.dot(y_hat - y)) / (x.shape[0])
    return gradient_vector


@_args_are_valid_arrays
def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
         alpha: float, max_iter: int) -> np.ndarray | None:
    """ Fits the model to the training dataset contained in x and y.

    Args:
        x (numpy.ndarray): matrix m * n, training examples, m = num of examples, 
                                                            n = num of features
        y (numpy.ndarray): vector m * 1, predicted values
        theta (numpy.ndarray): vector (n + 1) * 1, our parameters
        alpha (float): the learning rate
        max_iter (int): has to be positive, number of iterations done during gradiant descent

    Return:
        new_theta (numpy.ndarray): vector (n + 1) * 1, our new parameters
        None if any parameters is not of the required dimensions or type.
    """
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if not 0 <= alpha <= 1 or max_iter < 1:
        return None
    # Add column of 1s left of Matrix so we can vectorize the equation
    x = np.c_[np.ones((x.shape[0], 1)), x]
    # Check vectors are of valid dimension and compatible shape with matrix x
    m, n = x.shape
    if y.shape not in [(m, 1), (m, )] or theta.shape not in [(n, 1), (n, )]:
        return None
    for _ in range(max_iter):
        # Get gradient vector containing all the partial derivatives for each parameters
        # then update theta (parameters) based on our learning rate (alpha) and the gradient
        theta -= alpha * gradient(x, y, theta)
    return theta


def tests():
    """ Little test function """
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                 [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    expected = np.array([[41.99888822],
                         [0.97792316],
                         [0.77923161],
                         [-1.20768386]])
    np.testing.assert_array_almost_equal(theta2, expected)

    # Example 1:
    result = predict_(x, theta2)
    expected = np.array([[19.59925884],
                         [-2.80037055],
                         [-25.19999994],
                         [-47.59962933]])
    np.testing.assert_array_almost_equal(result, expected)

    print("All tests passing, no assert raised, gg")


if __name__ == "__main__":
    tests()
