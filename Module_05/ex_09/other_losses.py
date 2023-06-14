""" Deepen the notion of loss function in machine learning """
import numpy as np
# Needed for tests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def _args_are_valid(y, y_hat) -> bool:
    """Make sure the parameters are valid for our program

    Args:
        y (np.ndarray): vector of dimension m * 1
        y_hat (np.ndarray): vector of dimension m * 1

    Returns:
        bool: True if y and y_hat are of the desired type and dimensions, False otherwise
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


def mse_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    """ Calculate the MSE between the predicted output and the real output.
    Errors increase in quadratic fashion > penalizes errors, susceptible to outliers.

    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.

    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.

    Raises:
        This function should not raise any Exceptions.
    """
    if not _args_are_valid(y, y_hat):
        return None
    elem_num = len(y)
    mse = np.sum(np.square(y_hat - y)) / elem_num
    return mse


def rmse_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    """ Calculate the RMSE between the predicted output and the real output.
    Root of MSE.

    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.

    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.

    Raises:
        This function should not raise any Exceptions.
    """
    # Root of MSE
    rmse = mse_(y, y_hat) ** 0.5
    return rmse


def mae_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    """ Calculate the MAE between the predicted output and the real output.
    Errors increase in proportional fashion.

    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.

    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.

    Raises:
        This function should not raise any Exceptions.
    """
    # MSE but without the square
    if not _args_are_valid(y, y_hat):
        return None
    elem_num = len(y)
    mae = np.sum(np.abs(y_hat - y)) / elem_num
    return mae


def r2score_(y: np.ndarray, y_hat: np.ndarray) -> float | None:
    """ Calculate the R2score between the predicted output and the output.

    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.

    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.

    Raises:
        This function should not raise any Exceptions.
    """
    if not _args_are_valid(y, y_hat):
        return None
    # Independant variables / residual sum of squares -> sum of the difference between
    # the predicted values (y_hat) and the observed values (y)
    residual = np.sum((y_hat - y) ** 2)
    # Dependant variables / total sum of squares -> sum of the difference between the
    # observed values and the average of observed values
    total = np.sum((y - np.mean(y)) ** 2)
    r2score = 1 - (residual / total)
    return r2score


def main():
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    print("MSE")
    print(f"Result: {mse_(x, y)}")
    print(f"Expected: {mean_squared_error(x, y)}")
    print()

    # Root mean squared error
    print("RMSE")
    print(f"Result: {rmse_(x, y)}")
    print(f"Expected: {sqrt(mean_squared_error(x, y))}")
    print()

    # Mean absolute error
    print("MAE")
    print(f"Result: {mae_(x, y)}")
    print(f"Expected: {mean_absolute_error(x, y)}")

    print()

    # R2-score
    print("R2-score")
    r2score_(x, y)
    r2_score(x, y)
    print(f"Result: {r2score_(x, y)}")
    print(f"Expected: {r2_score(x, y)}")


if __name__ == "__main__":
    main()
