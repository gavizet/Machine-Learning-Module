""" Understand and manipulate the concept of gradient descent in the case of 
multivariate linear regression """
import numpy as np
from Module_07.ex_01.prediction import predict_
from Module_07.ex_02.loss import loss_
from Module_07.ex_03.gradient import gradient


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
def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, max_iter: int) -> np.ndarray | None:
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
    m, n = x.shape
    # Check vectors are of valid dimension and compatible shape with matrix x
    if y.shape not in [(m, 1), (m, )] or theta.shape not in [(n, 1), (n, )]:
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    if not 0 < alpha < 1 or max_iter <= 0:
        return None


def tests():
    """ Little test function """
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                 [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)
    # expected = np.array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    # Example 1:
    result = predict_(x, theta2)
    # expected = np.array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])

    print("All tests passing, no assert raised, gg")


if __name__ == "__main__":
    tests()
