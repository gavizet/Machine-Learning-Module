""" Write a class that contains all methods necessary to perform linear regression. """
import numpy as np


class MyLinearRegression():
    """ My personnal linear regression class to fit like a boss. """

    @staticmethod
    def _args_are_valid_ndarrays(*args) -> bool:
        """ Make sure the parameters are numpy ndarrays of valid dimensions for our program

        Args:
            *args (np.ndarray): vectors of dimension m * 1

        Returns:
            bool: True if args are of the desired type and dimensions, False otherwise
        """
        args_num = len(args)
        if args_num > 2:
            return False
        if args_num == 2 and args[0].shape != args[1].shape:
            return False
        for arg in args:
            if not isinstance(arg, np.ndarray):
                return False
            if not (np.issubdtype(arg.dtype, np.integer) or np.issubdtype(arg.dtype, np.floating)):
                return False
            if arg.size == 0 or arg.shape not in [(arg.size, ), (arg.size, 1)]:
                return False
        return True

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(alpha, float) or not 0 <= alpha <= 1:
            raise ValueError("alpha needs to be a float between 0 and 1.")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter needs to be a positive integer.")
        if not isinstance(thetas, np.ndarray) or \
                thetas.shape not in [(thetas.size, ), (thetas.size, 1)]:
            raise ValueError(
                "thetas has to be a numpy ndarray of dimension m * 1.")
        if not (np.issubdtype(thetas.dtype, np.integer) or
                np.issubdtype(thetas.dtype, np.floating)):
            raise ValueError(
                "thetas has to be a numpy ndarray of floats or ints")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def predict_(self, x: np.ndarray) -> np.ndarray | None:
        if not self._args_are_valid_ndarrays(x):
            return None
        return True

    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
        if not self._args_are_valid_ndarrays(y, y_hat):
            return None
        return True

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
        if not self._args_are_valid_ndarrays(y, y_hat):
            return None
        return True

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        if not self._args_are_valid_ndarrays(x, y):
            return None
        return True


def main():
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    mlr = MyLinearRegression(np.array([[2], [0.7]]))
    print(mlr.predict_(x))
    print(mlr.loss_elem_(x, y))
    print(mlr.loss_(x, y))
    print(mlr.fit_(x, y))


if __name__ == "__main__":
    main()
