import pytest
from Module_05.ex_00.matrix import Matrix, Vector

MATRIX_INIT = [
    (1, ValueError),  # Error : not a nested list
    ("test", ValueError),  # Error : not a nested list
    ([1, 2, 3], ValueError),  # Error : not a nested list of nums
    ([['a', 'b', 'c'], ['d', 'e', 'f']], ValueError),  # Error : not a nested list
    ([["test", 1.0], [2.0, 3.0]], ValueError),  # Error : not int/float
    ([[1, 2, 3], [4, 5]], ValueError),  # Bad shape
    (('a', 'b'), ValueError),  # Not a tuple of nums
    ((-1, 3), ValueError),  # Negative num
    ((1, 2, 3), ValueError),  # More than 2 nums
    ([[1], [2], [3]], [[[1], [2], [3]], (3, 1)]),  # Valid 1 row
    ([[1, 2, 3]], [[[1, 2, 3]], (1, 3)]),  # Valid 1 col
    ([[1, 2, 3], [4, 5, 6]], [[[1, 2, 3], [4, 5, 6]], (2, 3)]),  # Valid Matrix
    ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [
     [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], (2, 3)]),  # Valid Matrix
    ((1, 3), [[[0.0, 0.0, 0.0]], (1, 3)]),  # Valid tuple
    ((3, 3), [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], (3, 3)]),
]


@pytest.mark.parametrize("args, expected", MATRIX_INIT)
def test_matrix_init(args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            matrix = Matrix(args)
    else:
        matrix = Matrix(args)
        assert matrix.data == expected[0]
        assert matrix.shape == expected[1]


MATRIX_ADD_SUB = [
    (),
]

MATRIX_MUL_DIV = [
    (),
]

MATRIX_TRANSPOSE = [
    (),
]

VECTOR_INIT = [
    ("stupid test", ValueError),  # Error string
    (-3, ValueError),  # Error negative int
    ((10, 6), ValueError),  # Error tuple with a > b
    ([0.0, 1.0, 2.0, 3.0], ValueError),  # Error list of floats
    ([[0.0, 1.0], [2.0, 3.0]], ValueError),  # Error > than 1 float/sublist
    ((6, 10), ValueError),  # Not a valid vector shape
    ([[]], [[[]], (1, 0)]),  # Valid, gives empty values
    ((1, 3), [[[0.0, 0.0, 0.0]], (1, 3)]),  # Valid, filled with 0.0 values
    ([[1.0, 2.0, 3.0]], [[[1.0, 2.0, 3.0]], (1, 3)]),  # Valid row
    ([[1.0], [2.0], [3.0]], [[[1.0], [2.0], [3.0]], (3, 1)]),  # Valid column
]


@pytest.mark.parametrize("args, expected", VECTOR_INIT)
def test_vector_init(args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            vector = Vector(args)
    else:
        vector = Vector(args)
        assert vector.data == expected[0]
        assert vector.shape == expected[1]


VECTOR_ADD_SUB = [
    (),
]

VECTOR_MUL_DIV = [
    (),
]

VECTOR_DOT = [
    (),
]
