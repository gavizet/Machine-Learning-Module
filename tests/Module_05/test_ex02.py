import pytest
import numpy as np
from Module_05.ex_02.prediction import simple_predict

features = np.arange(1, 6)

THETA = [
    (np.array([5, 0]), np.array([5., 5., 5., 5., 5.])),
    (np.array([0, 1]), np.array([1., 2., 3., 4., 5.])),
    (np.array([5, 3]), np.array([8., 11., 14., 17., 20.])),
    (np.array([-3, 1]), np.array([-2., -1., 0., 1., 2.])),
]


@pytest.mark.parametrize("theta, expected", THETA)
def test_simple_predict(theta, expected):
    np.testing.assert_array_equal(simple_predict(features, theta), expected)


ERRORS = [
    (features, [1., 2., 3.], None),  # arg not nparray
    (features, np.array(["1", "2"]), None),  # arg not numbers
    (np.array([[1, 2], [1, 2]]), np.array([0, 1]), None),  # feature dim not 1
    (features, np.array([0, 1, 2]), None),  # theta shape not (2, 1)
    # (, , None),
]


@pytest.mark.parametrize("features, theta, expected", ERRORS)
def test_simple_predict_errors(features, theta, expected):
    result = simple_predict(features, theta)
    assert result == None
