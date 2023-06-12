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


m_1_3 = [[1, 2, 3]]
m_3_1 = [[1], [2], [3]]
m_3_2 = [[1, 2], [3, 4], [5, 6]]
m_2_3 = [[1, 2, 3], [4, 5, 6]]
m_3_3_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
m_3_3_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

MATRIX_ADD_SUB = [
    (Matrix(m_2_3), Vector(m_1_3), ValueError),  # Not same shape
    (Matrix(m_2_3), Matrix(m_3_2), ValueError),  # Not same shape
    (2, Matrix(m_2_3), ValueError),  # 1st arg not valid type
    (Matrix(m_2_3), 2, ValueError),  # 2nd arg not valid type
    (Matrix(m_3_3_a), Matrix(m_3_3_b), [
     [[10, 10, 10], [10, 10, 10], [10, 10, 10]], (3, 3)]),  # Valid matrix
    (Matrix(m_1_3), Matrix(m_1_3), [[[2, 4, 6]], (1, 3)]),  # Valid matrix
    (Vector(m_1_3), Vector(m_1_3), [[[2, 4, 6]], (1, 3)]),  # Valid vector
    (Vector(m_3_1), Vector(m_3_1), [[[2], [4], [6]], (3, 1)]),  # Valid vector
]


# Sub is same as add, no point having different tests
@pytest.mark.parametrize("arg1, arg2, expected", MATRIX_ADD_SUB)
def test_matrix_add_sub(arg1, arg2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = arg1 + arg2
    else:
        result = arg1 + arg2
        assert result.data == expected[0]
        assert result.shape == expected[1]


MATRIX_MUL = [
    (Matrix(m_2_3), Matrix(m_2_3), ValueError),  # n_col(A) != n_row(B)
    (Vector(m_3_1), Matrix(m_3_3_a), ValueError),  # n_col(A) != n_row(B)
    (Matrix(m_3_1), Matrix(m_3_1), ValueError),  # Not same shape
    # Valid Scalar * Matrix
    (2, Matrix(m_2_3), [[[2, 4, 6], [8, 10, 12]], (2, 3)]),
    # Valid Matrix * Scalar
    (Matrix(m_2_3), 2, [[[2, 4, 6], [8, 10, 12]], (2, 3)]),
    (Matrix(m_1_3), Vector(m_3_1), [[[14]], (1, 1)]),  # Valid Vector x Vector
    (Matrix(m_3_1), Vector(m_1_3), [
     [[1, 2, 3], [2, 4, 6], [3, 6, 9]], (3, 3)]),  # Valid Vector x Vector
    # Valid Matrix * Vector
    (Matrix(m_3_3_a), Vector(m_3_1), [[[14], [32], [50]], (3, 1)]),
    # Valid Matrix * Matrix
    (Matrix(m_2_3), Matrix(m_3_2), [[[22, 28], [49, 64]], (2, 2)]),
    (Matrix(m_3_2), Matrix(m_2_3), [
     [[9, 12, 15], [19, 26, 33], [29, 40, 51]], (3, 3)]),  # Valid Matrix * Matrix
    (Matrix(m_3_3_a), Matrix(m_3_3_b), [
     [[30, 24, 18], [84, 69, 54], [138, 114, 90]], (3, 3)]),  # Valid Matrix * Matrix
    (Matrix(m_3_3_b), Matrix(m_3_3_a), [
     [[90, 114, 138], [54, 69, 84], [18, 24, 30]], (3, 3)]),  # Valid Matrix * Matrix
]


@pytest.mark.parametrize("arg1, arg2, expected", MATRIX_MUL)
def test_matrix_mul(arg1, arg2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = arg1 * arg2
    else:
        result = arg1 * arg2
        assert result.data == expected[0]
        assert result.shape == expected[1]


MATRIX_DIV = [
    (Matrix(m_3_3_a), Matrix(m_3_3_a), ValueError),  # Not int/float div
    (Matrix(m_3_3_a), Vector(m_1_3), ValueError),  # Not int/float div
    (Matrix(m_3_3_a), "test", ValueError),  # Not int/float div
    (42, Matrix(m_1_3), ArithmeticError),  # Not implemented
    (Matrix(m_3_3_a), 2, [
     [[0.5, 1, 1.5], [2, 2.5, 3], [3.5, 4, 4.5]], (3, 3)]),  # Valid Matrix / Scalar
    (Vector(m_1_3), 2, [[[0.5, 1, 1.5]], (1, 3)]),  # Valid Vector / Scalar
]


@pytest.mark.parametrize("arg1, arg2, expected", MATRIX_DIV)
def test_matrix_div(arg1, arg2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = arg1 / arg2
    else:
        result = arg1 / arg2
        assert result.data == expected[0]
        assert result.shape == expected[1]


MATRIX_TRANSPOSE = [
    (Matrix(m_1_3), [[[1], [2], [3]], (3, 1)]),
    (Matrix(m_1_3), [[[1], [2], [3]], (3, 1)]),
    (Vector(m_3_1), [[[1, 2, 3]], (1, 3)]),
    (Vector(m_3_1), [[[1, 2, 3]], (1, 3)]),
    (Matrix(m_2_3), [[[1, 4], [2, 5], [3, 6]], (3, 2)]),
    (Matrix(m_3_2), [[[1, 3, 5], [2, 4, 6]], (2, 3)]),
    (Matrix(m_3_3_a), [[[1, 4, 7], [2, 5, 8], [3, 6, 9]], (3, 3)]),
]


@pytest.mark.parametrize("arg1, expected", MATRIX_TRANSPOSE)
def test_matrix_transpose(arg1, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = arg1.T()
    else:
        result = arg1.T()
        assert result.data == expected[0]
        assert result.shape == expected[1]


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


VECTOR_DOT = [
    (Vector((1, 4)), 4, ValueError),  # Error, can only dot 2 vectors
    (Vector((1, 4)), Vector((4, 1)), ValueError),  # Error, diff shape vectors
    (Matrix(m_1_3), Matrix(m_1_3), AttributeError),  # Matrix no dot method
    (Vector(m_1_3), Vector(m_1_3), 14),  # Valid dot product
    (Vector(m_3_1), Vector(m_3_1), 14),  # Valid dot product
]


@pytest.mark.parametrize("arg1, arg2, expected", VECTOR_DOT)
def test_vector_dot(arg1, arg2, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = arg1.dot(arg2)
    else:
        result = arg1.dot(arg2)
        assert result == expected

# Vector operations included in Matrix tests since a Matrix is just multiple vectors
# Thus, no need to test the operations separately outside of the dot product method,
# which is specific to vector in this exercice, and the __init__
