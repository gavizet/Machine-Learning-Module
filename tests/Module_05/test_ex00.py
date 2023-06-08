import pytest
from Module_05.ex_00.matrix import Matrix, Vector

MATRIX_INIT = [
    (Matrix(), ""),
]

@pytest.mark.parametrize("args, expected", MATRIX_INIT)
def test_matrix_init(args, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            matrix = Matrix(args)
    else:
        matrix = Matrix(args)
        assert matrix.values == expected[0]
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
    (),
]

VECTOR_ADD_SUB = [
    (),
]

VECTOR_MUL_DIV = [
    (),
]

VECTOR_DOT = [
    (),
]

@pytest.mark.parametrize("")
