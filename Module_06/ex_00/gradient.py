""" Understand and manipulate the notion of gradient and gradient descent in machine learning. """
import numpy as np


def _args_are_valid(x, y, theta):
    """Make sure the parameters are valid for our program

    Args:
        x (np.ndarray): vector of dimension m * 1
        y (np.ndarray): vector of dimension m * 1
        theta (np.ndarray): vector of dimension 2 * 1

    Returns:
        bool: True if x, y and theta are of the desired type and dimensions, False otherwise
    """
    params = [x, y, theta]
    if not all([isinstance(param, np.ndarray) for param in params]):
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
    return True


def simple_gradient(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
    """ Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.

    Args:
        x (numpy.array):, vector of shape m * 1, represents input values
        y (numpy.array): vector of shape m * 1, represents real values
        theta (numpy.array): a 2 * 1 vector, represents our parameters

    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.

    Raises:
        This function should not raise any Exception.
    """
    if not _args_are_valid(x, y, theta):
        return None
    elem_num = len(x)
    ones_col = np.ones((x.shape[0], 1))
    # Add column of 1 left of x so we can do the predict computation in one operation.
    x_prime = np.c_[ones_col, x]
    # Do the dot product between x_prime (input values) and y (real values)
    # to obtain y_hat (predicted values)
    y_hat = x_prime.dot(theta)
    # Create gradiant array of size 2 x 1
    gradiant = np.zeros((2, 1))
    # param 1 is just scaled
    gradiant[0] = np.sum((y_hat - y)) / elem_num
    # param 2 is scaled and multiplied by x
    gradiant[1] = np.sum((y_hat - y) * x) / elem_num
    return gradiant


def main():
    # x -> predicted values, y -> real values
    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))

    # Example 1:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(f"With a = {theta1[0]} and b = {theta1[1]}")
    print(f"Result: {repr(simple_gradient(x, y, theta1))}")
    print("Expected: array([[-19.0342574], [-586.66875564]])")
    print()

    # Example 2:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(f"With a = {theta2[0]} and b = {theta2[1]}")
    print(f"Result: {repr(simple_gradient(x, y, theta2))}")
    print("Expected: array([[-57.86823748], [-2230.12297889]])")
    print()

    # Example 3:
    theta3 = np.array([3, 2]).reshape((-1, 1))
    print(f"With a = {theta3[0]} and b = {theta3[1]}")
    print(f"Result: {repr(simple_gradient(x, y, theta3))}")
    print("Expected: array([[26.67862814], [1349.34177596]])")

    # Example 3:
    theta4 = np.array([2.4, 1.1]).reshape((-1, 1))
    print(f"With a = {theta4[0]} and b = {theta4[1]}")
    print(f"Result: {repr(simple_gradient(x, y, theta4))}")
    print("Expected: array([[-4.87644647], [12.201672]])")


if __name__ == "__main__":
    main()
