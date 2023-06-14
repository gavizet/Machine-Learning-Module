""" Understand and manipulate the notion of loss function in machine learning """
import numpy as np


def _args_are_valid(y, y_hat) -> bool:
    """Make sure the parameters are valid for our program

    Args:
        y - features (np.ndarray): vector of dimension m * 1
        y_hat - theta (np.ndarray): vector of dimension m * 1

    Returns:
        bool: True if features are of the desired type and dimensions, False otherwise
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return False
    if y.size == 0 or y.shape != y_hat.shape:
        return False
    if (not np.issubdtype(y.dtype, np.floating) and
            not np.issubdtype(y.dtype, np.integer)):
        return False
    if (not np.issubdtype(y_hat.dtype, np.floating) and
            not np.issubdtype(y_hat.dtype, np.integer)):
        return False
    if y.shape not in [(y.size, ), (y.size, 1)]:
        return False
    return True


def loss_(y, y_hat):
    """ Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.

    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.

    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.

    Raises:
        This function should not raise any Exceptions.
    """
    if not _args_are_valid(y, y_hat):
        return None
    elem_num = len(y)
    J_value = np.sum(np.square(y_hat - y)) / (elem_num * 2)
    return J_value


def main():
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    # Example 1:
    print(f"Result: {loss_(X, Y)}")
    print("Expected: 2.142857142857143")
    print("===============")

    # Example 2:
    print(f"Result: {loss_(X, X)}")
    print("Expected: 0.0")
    print("===============")

    # Example 3:
    print(f"Result: {loss_(Y, Y)}")
    print("Expected: 0.0")
    print("===============")

    # Example 4
    X = np.array([[34], [37], [44], [47], [48], [48],
                 [46], [43], [32], [27], [26], [24]])
    Y = np.array([[37], [40], [46], [44], [46], [50],
                 [45], [44], [34], [30], [22], [23]])
    print(f"Result: {loss_(X, Y)}")
    print("Expected: 2.9583333333333335")
    print("===============")


if __name__ == "__main__":
    main()

# Pytest test file with complete test battery can be found in tests/Module_05/test_ex07.py
