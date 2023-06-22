import numpy as np


def _args_are_valid_arrays(function):
    """ Little generator for error handling """
    def wrapper(*args, **kwargs):
        """ Wrapper function to make sure the parameters are 
            numpy ndarrays of valid dimensions and type

        Args:
            *args (numpy.ndarray): vectors of dimension m * 1

        Returns:
            bool: function if args are of the desired type and dimensions, None otherwise
        """
        for arg in args:
            if not isinstance(arg, np.ndarray):
                return None
            if not (np.issubdtype(arg.dtype, np.integer) or np.issubdtype(arg.dtype, np.floating)):
                return None
            if arg.size == 0:
                return None
        return function(*args, **kwargs)
    return wrapper

@_args_are_valid_arrays
def loss_(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
    """ Computes the mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.

    Args:
        y (numpy.ndarray): vector of shape (m, 1), the real values
        y_hat: has to be an numpy.array, a vector, the predicted values

    Return:
        mse (float): float, the mean squared error of the two vectors.
        None if y or y_hat are not of the required dimensions or type.
    """
    if y.shape != y_hat.shape or 
    pass
