""" You must implement a function to plot the data and the regression line. """
import numpy as np
import matplotlib.pyplot as plt
from Module_05.ex_04.prediction import predict_


def _ndarray_is_int_float(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.floating) and not np.issubdtype(array.dtype, np.integer):
        return False
    return True


def _args_are_valid(x, y, theta) -> bool:
    args = [x, y, theta]
    for arg in args:
        if not isinstance(arg, np.ndarray) or not _ndarray_is_int_float(arg):
            return False
    if x.shape not in [(x.size, ), (x.size, 1)]:
        return False
    if theta.shape not in [(2, ), (2, 1)]:
        return False
    if x.shape != y.shape:
        return False
    return True


def plot(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
    """ Plot the data and prediction line from three non-empty numpy.array.

    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.

    Returns:
        Nothing.

    Raises:
        This function should not raise any Exceptions.
    """
    if not _args_are_valid(x, y, theta):
        return None
    prediction = predict_(x, theta)
    plt.figure()
    plt.scatter(x, y, marker='o')
    plt.plot(x, prediction, color='orange')
    plt.show()


def main():
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    # Example 1:
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    # Example 2:
    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    # Example 3:
    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)


if __name__ == "__main__":
    main()
