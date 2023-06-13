import pytest
import numpy as np
from Module_05.ex_04.prediction import predict

PARAMS = [
    (),
]


@pytest.mark.parametrize("x, theta, expected", PARAMS)
def test_add_intercept(x, theta, expected):
    np.testing.assert_array_equal(predict(x, theta), expected)


ERRORS = [
    (),
]


@pytest.mark.parametrize("x, theta, expected", ERRORS)
def test_simple_predict_errors(x, theta, expected):
    result = predict(x, theta)
    assert result == expected
