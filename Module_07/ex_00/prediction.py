import numpy as np


def _args_are_valid_arrays(x, theta) -> bool:
    """ Make sure the parameters are numpy ndarrays of valid dimensions for our program

    Args:
        x (numpy.array): matrix of dimension m * n, the training examples
        theta (numpy.array): vector of dimension (n + 1) * 1, the parameters

    Returns:
        bool: True if args are of the desired type and dimensions, False otherwise
    """
    # Check x Matrix has the same number of columns as the theta vector has rows.
    if x.shape[0] != theta.size:
        return False
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return False
    if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        return False
    if not (np.issubdtype(theta.dtype, np.integer) or np.issubdtype(theta.dtype, np.floating)):
        return False
    if x.size == 0 or theta.size == 0:
        return False
    if theta.shape not in [(theta.size, ), (theta.size, 1)]:
        return False
    return True


def simple_predict(x, theta) -> np.ndarray | None:
    """ Computes the prediction vector y_hat from the design matrix (x) and the parameters (theta).

    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.

    Return:
        y_hat (numpy.array) : vector of dimension m * 1, represents our predictions
        None if x or theta are not of the required dimensions or type.
    """
    if not _args_are_valid_arrays(x, theta):
        return None
    y_hat = x.dot(theta)
    return y_hat


def main():
    x = np.arange(1, 13).reshape((4, -1))
    print(x)

    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    simple_predict(x, theta1)
    # Ouput: array([[5.], [5.], [5.], [5.]])
    # Do you understand why y_hat contains only 5’s here?

    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    simple_predict(x, theta2)
    # Output: array([[ 1.], [ 4.], [ 7.], [10.]])
    # Do you understand why y_hat == x[:,0] here?

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    simple_predict(x, theta3)
    # Output: array([[ 9.64], [24.28], [38.92], [53.56]])

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    simple_predict(x, theta4)
    # Output: array([[12.5], [32. ], [51.5], [71. ]])


if __name__ == "__main__":
    main()
