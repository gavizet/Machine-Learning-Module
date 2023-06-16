import numpy as np


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
    return "Yo"


def main():
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    # Output:
    # array([[1.40709365],
    # [1.1150909 ]])

    # Example 1:
    predict(x, theta1)
    # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])


if __name__ == "__main__":
    main()
