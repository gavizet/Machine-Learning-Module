""" Write a class that contains all methods necessary to perform linear regression. """
import numpy as np


class MyLinearRegression():
    """ My personnal multvivariate linear regression class. """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(alpha, float) or not 0 <= alpha <= 1:
            raise ValueError("alpha needs to be a float between 0 and 1.")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter needs to be a positive integer.")
        if not isinstance(thetas, np.ndarray) or \
                thetas.shape not in [(thetas.size, ), (thetas.size, 1)]:
            raise ValueError(
                "thetas has to be a numpy.ndarray of dimension m * 1.")
        if not (np.issubdtype(thetas.dtype, np.integer) or
                np.issubdtype(thetas.dtype, np.floating)):
            raise ValueError(
                "thetas has to be a numpy.ndarray of floats or ints")
        self.alpha = alpha
        self.max_iter = max_iter
        # Cast as float64, otherwise numpy throws a _UFuncOutputCastingError during fit_
        self.thetas = thetas.astype('float64')

    @staticmethod
    def _args_are_valid_ndarrays(*args) -> bool:
        """ Make sure the parameters are numpy ndarrays of valid dimensions for our program

        Args:
            *args (numpy.ndarray): vectors of dimension m * 1

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

    def _args_are_valid_arrays(self, function):
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
                if not (np.issubdtype(arg.dtype, np.integer) or
                        np.issubdtype(arg.dtype, np.floating)):
                    return None
                if arg.size == 0:
                    return None
            return function(self, *args, **kwargs)
        return wrapper

    @_args_are_valid_arrays
    def add_intercept_(self, x: np.ndarray) -> np.ndarray | None:
        """ Adds a column of '1' to the x numpy.ndarray.
        Args:
            x (numpy.ndarray): has to be of dimension m * n.

        Returns:
            x_prime (numpy.array): of dimension m * (n + 1).
            None if any argument is not of the expected type or dimensions.
        """
        # Create column filled with 1s with same size as x
        ones_col = np.ones((x.shape[0], 1))
        # Add column left of vector x so it becomes a Matrix X'
        x_prime = np.c_[ones_col, x]
        return x_prime

    @_args_are_valid_arrays
    def predict_(self, x: np.ndarray) -> np.ndarray | None:
        """ Computes the vector of prediction y_hat from two non-empty numpy.array 
            representing our input values (x) and our parameters (theta).

        Args:
            x (np.ndarray): vector of dimension m * 1, represents our input values.

        Returns:
            y_hat (np.ndarray): vector of dimension m * 1, represents our expected values.
            None if any argument is not of the expected type or dimensions.
        """
        x_prime = self.add_intercept_(x)
        # Do dot product between X' (input values) and theta (params) to get our expected values
        y_hat = x_prime.dot(self.thetas)
        return y_hat

    @_args_are_valid_arrays
    def loss_elem_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
        """ Computes all the elements (y_pred - y)^2 of the loss function.

        Args:
            y (numpy.ndarray): vector of dimensions m * 1.
            y_hat (numpy.ndarray): vector of dimensions m * 1

        Returns:
            loss_elem (numpy.ndarray): vector of dimension m * 1.
            None if any argument is not of the expected type or dimensions.
        """
        elem_loss = np.square(y_hat - y)
        return elem_loss

    @_args_are_valid_arrays
    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
        """ Computes the half mean squared error between y_hat and y

        Args:
            y (np.ndarray): vector (m, 1), the real values.
            y_hat (np.ndarray): vector (m, 1), the expected values.

        Returns:
            loss : float, the lower it is, the better
            None if any argument is not of the expected type or dimensions.
        """
        elem_num = len(y)
        # Sum the loss for each expected value and divide by the number of values * 2
        # to get our loss number
        loss = np.sum(self.loss_elem_(y, y_hat)) / (elem_num * 2)
        return loss

    @_args_are_valid_arrays
    def mse_(self, y: np.ndarray, y_hat: np.ndarray) -> float | None:
        """ Calculate the MSE between the predicted output and the real output.
        Errors increase in quadratic fashion > penalizes errors, susceptible to outliers.

        Args:
            y (numpy.array): vector of dimension m * 1, real values
            y_hat (numpy.ndarray): vector of dimension m * 1, expected value

        Returns:
            mse: has to be a float. 
            None if any argument is not of the expected type or dimensions.
        """
        elem_num = len(y)
        mse = np.sum(self.loss_elem_(y, y_hat)) / elem_num
        return mse

    @_args_are_valid_arrays
    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """ Computes a gradient vector from two non-empty numpy.array.

        Args:
            x (numpy.array): vector of shape m * 1.
            y (numpy.ndarray): vector of shape m * 1.

        Return:
            gradient (numpy.ndarray): vector of the same dimensions as self.thetas.
            None if any argument is not of the expected type or dimensions.
        """
        elem_num = len(x)
        # Get predicted values
        y_hat = self.predict_(x)
        # Add column to input values x to make it a matrix,
        # then Transpose so we can multiply with (y_hat - y).
        x_prime_transpose = np.transpose(self.add_intercept_(x))
        # Get gradient so we know in which direction we need to move each theta
        gradient = (x_prime_transpose.dot(y_hat - y)) / elem_num
        return gradient

    @_args_are_valid_arrays
    def fit_(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Fits the model to the training dataset contained in x and y and
            update our self.thetas (params) based on the result

        Args:
            x (np.ndarray): vector of dimension m * 1, represents our predicted values.
            y (np.ndarray): vector of dimension m * 1, represents our real values.

        Returns:
            None if any argument is not of the expected type or dimensions.
        """
        for _ in range(self.max_iter):
            # Get gradiant for current value of thetas (params) then
            # update thetas based on alpha and the gradient value we just got
            self.thetas -= self.alpha * self.gradient_(x, y)


def tests():
    x = np.array([[12.4956442],
                  [21.5007972],
                  [31.5527382],
                  [48.9145838],
                  [57.5088733]])
    y = np.array([[37.4013816],
                  [36.1473236],
                  [45.7655287],
                  [46.6793434],
                  [59.5585554]])

    # Example 0.0:
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    y_hat = lr1.predict_(x)
    expected_yhat = np.array([[10.74695094],
                              [17.05055804],
                              [24.08691674],
                              [36.24020866],
                              [42.25621131]])
    np.testing.assert_array_almost_equal(y_hat, expected_yhat)

    # Example 0.1:
    result = lr1.loss_elem_(y, y_hat)
    expected = np.array([[710.45867381],
                        [364.68645485],
                        [469.96221651],
                        [108.97553412],
                        [299.37111101]])
    np.testing.assert_array_almost_equal(result, expected)

    # Example 0.2:
    result = lr1.loss_(y, y_hat)
    expected = 195.34539903032385
    np.testing.assert_almost_equal(result, expected)

    # Example 1.0:
    lr2 = MyLinearRegression(
        np.array([[1], [1]]), alpha=0.00001, max_iter=100000)
    lr2.fit_(x, y)
    thetas = lr2.thetas
    expected_thetas = np.array([[5.943245],
                                [1.008418]])
    np.testing.assert_array_almost_equal(thetas, expected_thetas)

    # Example 1.1:
    y_hat = lr2.predict_(x)
    expected = np.array([[18.544077],
                        [27.625036],
                        [37.761594],
                        [55.26959093],
                        [63.93622697]])
    np.testing.assert_array_almost_equal(y_hat, expected)

    # Example 1.2:
    result = lr2.loss_elem_(y, y_hat)
    expected = np.array([[355.597918],
                        [72.629391],
                        [64.062976],
                        [73.79235259],
                        [19.16400836]])
    np.testing.assert_array_almost_equal(result, expected)

    # Example 1.3:
    result = lr2.loss_(y, y_hat)
    expected = 58.524664606009345
    np.testing.assert_almost_equal(result, expected)

    print("No assert was raised, all tests passed. GG !")


if __name__ == "__main__":
    tests()
    # No pytest file for this one, it's exactly the same error handling as before
    # on top of it, program takes too long to run if we want to get any decent results.
