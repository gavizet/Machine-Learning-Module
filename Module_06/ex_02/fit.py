import numpy as np


def add_intercept_(x: np.ndarray) -> np.ndarray:
    """Adds a column of '1' to the non-empty numpy.array x.
    Args:
        array: has to be a numpy.array of dimension m * n.

    Returns:
        x_prime (numpy.array): of dimension m * (n + 1).

    Raises:
        This function should not raise any Exception.
    """
    ones_col = np.ones((x.shape[0], 1))
    x_prime = np.c_[ones_col, x]
    return x_prime


def predict_(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """ Computes the vector of prediction y_hat from two non-empty numpy.array 
        representing our input values (x) and our parameters (theta).

    Args:
        x (np.ndarray): vector of dimension m * 1, represents our input values.
        theta (np.ndarray): vector of dimension 2 * 1, represents our parameters.

    Returns:
        y_hat (np.ndarray): vector of dimension m * 1, represents our expected values.

    Raises:
        This function should not raise any Exceptions.
    """
    x_prime = add_intercept_(x)
    y_hat = x_prime.dot(theta)
    return y_hat


def gradient_(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """ Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.

    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.

    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.

    Raises:
        This function should not raise any Exception.
    """
    elem_num = len(x)
    # Get predicted values
    y_hat = predict_(x, theta)
    # Add column to input values x to make it a matrix,
    # then Transpose so we can multiply with (y_hat - y).
    x_prime_transpose = np.transpose(add_intercept_(x, ))
    # Get gradient so we know in which direction we need to move each theta
    gradient = (x_prime_transpose.dot(y_hat - y)) / elem_num
    return gradient


def _args_are_valid(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
                    alpha: float, max_iter: int) -> bool:
    """Make sure the parameters are valid for our program

    Args:
        x (np.ndarray): vector of dimension m * 1
        y (np.ndarray): vector of dimension m * 1
        theta (np.ndarray): vector of dimension 2 * 1
        alpha (float): the learning rate between 0 and 1.
        max_iter (int): the number of iterations, has to be positive.

    Returns:
        bool: True if x, y, theta, alpha and max_iter are of the desired type 
            and dimensions, False otherwise
    """
    params = [x, y, theta]
    if not all([isinstance(param, np.ndarray) for param in params]) or \
            not isinstance(alpha, float) or \
            not isinstance(max_iter, int):
        return False
    if not all([(np.issubdtype(param.dtype, np.floating) or
                 np.issubdtype(param.dtype, np.integer))
                for param in params]):
        return False
    if x.size == 0 or x.shape != y.shape:
        return False
    if x.shape not in [(x.size, ), (x.size, 1)] or \
            theta.shape not in [(theta.size, ), (theta.size, 1)]:
        return False
    if not 0 <= alpha <= 1 or max_iter < 1:
        return False
    return True


def fit_(x: np.ndarray, y: np.ndarray, theta: np.ndarray,
         alpha: float, max_iter: int) -> np.ndarray | None:
    """ Fits the model to the training dataset contained in x and y.

    Args:
        x (np.ndarray): vector of dimension m * 1, represents our predicted values.
        y (np.ndarray): vector of dimension m * 1, represents our real values.
        theta (np.ndarray): vector of dimension 2 * 1 representing the parameters.
        alpha (float): the learning rate between 0 and 1.
        max_iter (int): the number of iterations done during the gradient descent.

    Returns:
        new_theta (np.ndarray): vector of dimension 2 * 1. 
            Scaled parameters so that we get our loss function at its minimum value.
        None if there is a matching dimension problem.

    Raises:
        This function should not raise any Exception.
    """
    if not _args_are_valid(x, y, theta, alpha, max_iter):
        return None
    # Cast as float64, otherwise numpy throws a _UFuncOutputCastingError
    theta = theta.astype('float64')
    for _ in range(max_iter):
        gradient = gradient_(x, y, theta)
        theta -= alpha * gradient
    return theta


def main():
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(f"Result: {repr(theta1)}")
    print("Expected: array([[1.40709365], [1.1150909 ]])")
    np.testing.assert_array_almost_equal(theta1, np.array([[1.40709365],
                                                           [1.1150909]]))

    # Example 1:
    pred_result = predict_(x, theta1)
    print(f"Result: {repr(pred_result)}")
    print(
        "Expected: array([[15.3408728 ],\n [25.38243697],\n [36.59126492],\n [55.95130097],\n [65.53471499]]")
    np.testing.assert_array_almost_equal(pred_result, np.array([[15.3408728],
                                                                [25.38243697],
                                                                [36.59126492],
                                                                [55.95130097],
                                                                [65.53471499]]))


def generateSample(N, variance):
    X = np.array(range(N)).reshape((-1, 1))
    Y = X * variance
    theta = np.array([3, 6]).reshape((-1, 1))
    return X, Y, theta


def test(N, variance):
    x, y, theta = generateSample(N, variance)
    prediction = predict_(x, theta)
    print(f"Initial Predict: {repr(prediction)}")
    new_theta = fit_(x, y, theta, alpha=0.00000001, max_iter=150000)
    prediction = predict_(x, new_theta)
    print(f"After fit predict: {repr(prediction)}")


if __name__ == "__main__":
    main()
    test(10, 5)
    # No arg checking in predict_ and gradient_ because it is already checked in fit_
    # No pytest file for this one, it's exactly the same error handling as before
    # on top of it, program takes too long to run if we want to get any decent results.
    # Cba generating random test data as well :)
