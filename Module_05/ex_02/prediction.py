import numpy as np


def _args_are_valid(features, theta) -> bool:
    if not isinstance(features, np.ndarray) or not isinstance(theta, np.ndarray):
        return False
    print(features.shape)
    print(theta.shape)
    if features.shape[1] != 1 or theta.shape != (2, 1):
        return False
    return True


def simple_predict(features, theta) -> np.ndarray or None:
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.

    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.

    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.

    Raises:
        This function should not raise any Exception.
    """
    if not _args_are_valid(features, theta):
        return None
    return "Yo"


def main():
    features_vector = 3
    param_vector = 0
    result = simple_predict(features_vector, param_vector)
    print(result)


if __name__ == "__main__":
    main()
