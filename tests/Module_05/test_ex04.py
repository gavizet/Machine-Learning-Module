import pytest
import numpy as np
from Module_05.ex_04.prediction import predict_

feat = np.arange(1, 6)

THETA = [
    (np.array([5, 0]), np.array([5., 5., 5., 5., 5.])),
    (np.array([0, 1]), np.array([1., 2., 3., 4., 5.])),
    (np.array([5, 3]), np.array([8., 11., 14., 17., 20.])),
    (np.array([-3, 1]), np.array([-2., -1., 0., 1., 2.])),
]


@pytest.mark.parametrize("theta, expected", THETA)
def test_predict_(theta, expected):
    np.testing.assert_array_equal(predict_(feat, theta), expected)


ERRORS = [
    (feat, [1., 2., 3.], None),  # arg not nparray
    (feat, np.array(["1", "2"]), None),  # arg not numbers
    (np.array([[1, 2], [1, 2]]), np.array([0, 1]), None),  # feature dim not 1
    (feat, np.array([0, 1, 2]), None),  # theta shape not (2, 1)
]


@pytest.mark.parametrize("x, theta, expected", ERRORS)
def test_errors_predict_(x, theta, expected):
    result = predict_(x, theta)
    assert result == expected
