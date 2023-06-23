import numpy as np


class MyLinearRegression:
    """ My personnal multivariate linear regression """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if not isinstance(alpha, float) or not 0 <= alpha <= 1:
            raise ValueError("alpha needs to be a float between 0 and 1.")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise ValueError("max_iter needs to be a positive integer.")
        if not isinstance(thetas, np.ndarray) or thetas.size == 0:
            raise ValueError(
                "thetas has to be a numpy.ndarray of dimension m * 1.")
        if not (np.issubdtype(thetas.dtype, np.integer) or
                np.issubdtype(thetas.dtype, np.floating)):
            raise ValueError(
                "thetas has to be a numpy.ndarray of floats or ints")
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def predict_(self, x: np.ndarray) -> np.ndarray | None:
        """ Computes the prediction vector y_hat

        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            theta: has to be an numpy.array, a vector of dimension (n + 1) * 1.

        Return:
            y_hat (numpy.array) : vector of dimension m * 1, represents our predictions
            None if x or theta are not of the required dimensions or type.
        """
        x_prime = np.c_[np.ones((x.shape[0], 1)), x]
        y_hat = x_prime.dot(self.thetas)
        return y_hat

    def loss_(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray | None:
        """ Computes the half mean squared error between y and y_hat

        Args:
            y (numpy.ndarray): vector m * 1, the real values
            y_hat (numpy.ndarray): vector m * 1, the predicted values

        Return:
            mse (float): float, the mean squared error of the two vectors.
            None if y or y_hat are not of the required dimensions or type.
        """
        loss = np.sum(np.square(y_hat - y)) / (len(y_hat) * 2)
        return loss

    def gradient_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """ Computes a gradient vector.

        Args:
            x (numpy.ndarray): matrix m * n + 1, training examples, 
                                m = num of values, n = num of features
            y (numpy.ndarray): vector m * 1, predicted values
            theta (numpy.ndarray): vector (n + 1) * 1, our parameters

        Return:
            gradient (numpy.ndarray): vector (n + 1) * 1
            containg the result of the formula for all J (loss).
            None if y or y_hat are not of the required dimensions or type.
        """
        y_hat = x.dot(self.thetas)
        x_transpose = np.transpose(x)
        gradient_vector = (x_transpose.dot(y_hat - y)) / (x.shape[0])
        return gradient_vector

    def fit_(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """ Fits the model to the training dataset contained in x and y.

        Args:
            x (numpy.ndarray): matrix m * n, training examples, m = num of examples, 
                                                                n = num of features
            y (numpy.ndarray): vector m * 1, predicted values

        Return:
            self.thetas (numpy.ndarray): vector (n + 1) * 1, our new parameters
            None if any parameters is not of the required dimensions or type.
        """
        x = np.c_[np.ones((x.shape[0], 1)), x]
        m, n = x.shape
        if y.shape not in [(m, 1), (m, )] or self.thetas.shape not in [(n, 1), (n, )]:
            return None
        for _ in range(self.max_iter):
            self.thetas -= self.alpha * self.gradient_(x, y)
        return self.thetas


def tests():
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    theta_ex = np.array([[1.], [1.], [1.], [1.], [1.]])
    mylr = MyLinearRegression(theta_ex)

    # Example 0:
    y_hat = mylr.predict_(X)
    expected = np.array([[8.], [48.], [323.]])
    np.testing.assert_array_almost_equal(y_hat, expected)

    # Example 1:
    result = mylr.loss_(Y, y_hat)
    expected = 1875.0
    np.testing.assert_equal(result, expected)

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    result = mylr.thetas
    expected = np.array([[1.81883792e+01],
                         [2.76697788e+00],
                         [-3.74782024e-01],
                         [1.39219585e+00],
                         [1.74138279e-02]])
    np.testing.assert_array_almost_equal(result, expected)

    # Example 4:
    y_hat = mylr.predict_(X)
    expected = np.array([[23.41720822],
                        [47.48924883],
                        [218.06563769]])
    np.testing.assert_array_almost_equal(y_hat, expected)

    # Example 5:
    result = mylr.loss_(Y, y_hat)
    expected = 0.07320629376956866
    np.testing.assert_equal(result, expected)

    print("No assert raised, all tests passing, jayjay")


if __name__ == "__main__":
    tests()
# Assuming all given arguments are valid here, since we already tested all the error handling
# in previous exercices
