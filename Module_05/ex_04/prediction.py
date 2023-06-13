""" Understand and manipulate the notion of hypothesis in machine learning. """
import numpy as np
from Module_05.ex_03.tools import add_intercept


def _args_are_valid(features, theta) -> bool:
    """Make sure the parameters are valid for our program

    Args:
        features (np.ndarray): vector of dimension m * 1
        theta (np.ndarray): vector of dimension 2 * 1

    Returns:
        bool: True if features are of the desired type and dimensions, False otherwise
    """
    if not isinstance(features, np.ndarray) or not isinstance(theta, np.ndarray):
        return False
    if (not np.issubdtype(features.dtype, np.floating) and
            not np.issubdtype(features.dtype, np.integer)):
        return False
    if (not np.issubdtype(theta.dtype, np.floating) and
            not np.issubdtype(theta.dtype, np.integer)):
        return False
    if features.shape not in [(features.size, ), (features.size, 1)]:
        return False
    if theta.shape not in [(2, ), (2, 1)]:
        return False
    return True


def predict_(x, theta):
    """ Computes the vector of prediction y_hat from two non-empty numpy.array.

    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.

    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.

    Raises:
        This function should not raise any Exceptions.
    """
    if not _args_are_valid(x, theta):
        return None
    x_prime = add_intercept(x)
    y_hat = x_prime.dot(theta)
    return y_hat


def main():
    features_vector = np.arange(1, 6)

    # Test 1
    param_vector1 = np.array([5, 0])
    result = predict_(features_vector, param_vector1)
    print("=============")
    print("Expected: array([5., 5., 5., 5., 5.])")
    print(f"Result: {repr(result)}")

    # Test 2
    param_vector2 = np.array([0, 1])
    result = predict_(features_vector, param_vector2)
    print("=============")
    print("Expected: array([1., 2., 3., 4., 5.])")
    print(f"Result: {repr(result)}")

    # Test 3
    param_vector3 = np.array([5, 3])
    result = predict_(features_vector, param_vector3)
    print("=============")
    print("Expected: array([ 8., 11., 14., 17., 20.])")
    print(f"Result: {repr(result)}")

    # Test 4
    param_vector4 = np.array([-3, 1])
    result = predict_(features_vector, param_vector4)
    print("=============")
    print("Expected: array([-2., -1., 0., 1., 2.])")
    print(f"Result: {repr(result)}")


if __name__ == "__main__":
    main()


# Pytest test file can be found in tests/Module_05/test_ex04.py
