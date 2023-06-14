import numpy as np
from Module_05.ex_04.prediction import predict_


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


def loss_elem_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray or None:
    """ Calculates all the elements (y_pred - y)^2 of the loss function.

    Args:
        y (np.ndarray): a vector.
        y_hat (np.ndarray): a vector

    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.

    Raises:
        This function should not raise any Exception.
    """
    if not _args_are_valid(y, y_hat):
        return None
    J_elem = np.square(y_hat - y)
    return J_elem


def loss_(y: np.ndarray, y_hat: np.ndarray) -> float or None:
    """ Calculates the value of loss function.

    Args:
        y (np.ndarray): a vector.
        y_hat (np.ndarray): a vector

    Returns:
        J_value : float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.

    Raises:
        This function should not raise any Exception.
    """
    if not _args_are_valid(y, y_hat):
        return None
    # This is almost equivalent to the Mean Squared Error loss function, except we divide by 2
    elem_num = len(y)
    J_value = np.sum(loss_elem_(y, y_hat)) / (elem_num * 2)
    return J_value


def main():
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    # Example 1:
    print(f"Result: {repr(loss_elem_(y1, y_hat1))}")
    print(f"Expected: {repr(np.array([[0.], [1], [4], [9], [16]]))}")
    print("===============")

    # Example 2:
    print(f"Result: {loss_(y1, y_hat1)}")
    print("Expected: 3.0")
    print("===============")

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    # Example 3:
    print(f"Result: {loss_(y2, y_hat2)}")
    print("Expected: 2.142857142857143")
    print("===============")

    # Example 4: perfect model, no error
    print(f"Result: {loss_(y2, y2)}")
    print("Expected: 0.0")


if __name__ == "__main__":
    main()

# Pytest test file with complete test battery can be found in tests/Module_05/test_ex06.py
