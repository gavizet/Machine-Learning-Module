""" Manipulate the hypothesis to make prediction. """
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
def simple_predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray | None:
    """ Computes the prediction vector y_hat from the design matrix (x) and the parameters (theta).

    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.

    Return:
        y_hat (numpy.array) : vector of dimension m * 1, represents our predictions
        None if x or theta are not of the required dimensions or type.
    """
    if x.shape[1] + 1 != theta.size:
        return None
    # Add column of ones left of our matrix X so we can do the dot product between X and theta.
    # We essentially just multiply / scale theta_0 by 1 here, which does not change
    # anything to the hypothesis since the first element of the hypothesis is just theta_0.
    x_prime = np.c_[np.ones((x.shape[0], 1)), x]
    y_hat = x_prime.dot(theta)
    return y_hat


def main():
    x = np.arange(1, 13).reshape((4, -1))

    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    result = simple_predict(x, theta1)
    expected = np.array([[5.], [5.], [5.], [5.]])
    np.testing.assert_array_almost_equal(result, expected)
    # print(f"Result: {repr(result)}")
    # print(f"Expected: {repr(expected)}")

    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    result = simple_predict(x, theta2)
    expected = np.array([[1.], [4.], [7.], [10.]])
    np.testing.assert_array_almost_equal(result, expected)
    # print(f"Result: {repr(result)}")
    # print(f"Expected: {repr(expected)}")

    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    result = simple_predict(x, theta3)
    expected = np.array([[9.64], [24.28], [38.92], [53.56]])
    np.testing.assert_array_almost_equal(result, expected)
    # print(f"Result: {repr(result)}")
    # print(f"Expected: {repr(expected)}")

    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    result = simple_predict(x, theta4)
    expected = np.array([[12.5], [32.], [51.5], [71.]])
    np.testing.assert_array_almost_equal(result, expected)
    # print(f"Result: {repr(result)}")
    # print(f"Expected: {repr(expected)}")

    print("All tests passed without raising any assert, gg.")


if __name__ == "__main__":
    main()
