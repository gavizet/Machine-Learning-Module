import numpy as np
from Module_05.ex_04.prediction import predict_


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
    pass


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
    pass


def main():
    pass


if __name__ == "__main__":
    main()
