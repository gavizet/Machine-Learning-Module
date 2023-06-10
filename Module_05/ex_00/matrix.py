"""" Manipulation and understanding of basic matrix operations.
Numpy usage is forbidden"""
from typing import Self, List, Tuple


class Matrix:
    """" Matrix manipulation class"""

    @staticmethod
    def __check_args(arg) -> None:
        if not isinstance(arg, (List, Tuple)):
            raise ValueError(
                f"Argument has to be a List or a Tuple, not {type(arg)}")
        if isinstance(arg, List):
            for index, elem in enumerate(arg):
                if not isinstance(elem, List):
                    raise ValueError(f"Argumnt has to be a List of List, \
                         not a List of {type(elem)}")
                if index == 0:
                    len_elem = len(elem)
                elif len_elem != len(elem) or len(elem) == 0:
                    raise ValueError(
                        "All sublists need to have the same length (> 0)")
                if not all(isinstance(num, (int, float)) for num in elem):
                    raise ValueError(
                        "All elements of the sublists need to be an int or a float")
        if isinstance(arg, Tuple):
            if len(arg) != 2 or not \
                    all((isinstance(num, (int, float)) and num > 0) for num in arg):
                raise ValueError(
                    "Shape argument should be a Tuple of 2 positive int / float")

    @staticmethod
    def __get_data(arg) -> list:
        if isinstance(arg, list):
            data = arg
        else:
            # List of list of zeroes
            data = [[0.0] * arg[1]] * arg[0]
            return data
        return data

    @staticmethod
    def __get_shape(arg) -> tuple:
        if isinstance(arg, tuple):
            shape = arg
        else:
            # Tuple of 2 positive numbers
            shape = (len(arg), len(arg[0]))
        return shape

    def __init__(self, arg: list or tuple):
        self.__check_args(arg)
        self.data = self.__get_data(arg)
        self.shape = self.__get_shape(arg)
        self.type = type(self)
        print(self.type)

    def __add__(self, other: Self):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Can only add between 2 Matrix of the same shape")
        result = [[self.data[row][col] + other.data[row][col]
                   for col in range(self.shape[1])]
                  for row in range(self.shape[0])]
        return type(self)(result)

    def __radd__(self, other: Self):
        return self.__add__(other)

    def __sub__(self, other: Self):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Can only add between 2 Matrix of the same shape")
        result = [[self.data[row][col] - other.data[row][col]
                   for col in range(self.shape[1])]
                  for row in range(self.shape[0])]
        return type(self)(result)

    def __rsub__(self, other: Self):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            if isinstance(other, (int, float)):
                # Multiply each number of the Matrix by Scalar
                result = [[self.data[row][col] * other.data[row][col]
                           for col in range(self.shape[1])]
                          for row in range(self.shape[0])]
                return type(self)(result)
            if isinstance(other, (Matrix, Vector)):
                # Can only multiply if n_col(self) == n_row(other)
                if self.shape[1] != other.shape[0]:
                    raise ValueError("Wrong shape. Can only do AxB if the number of columns of \
                                    A is equal to the number of rows of B.")
                # Result Shape is always (n_row(self), n_col(other)).
                # Matrix x Matrix => Matrix || Matrix x Vector => Vector
                # B(3, 1) * A(1, 3) => C(3, 3) || A(1, 3) * B(3, 1) => C(1, 1)
                if self.shape[0] == 1 or other.shape[1] == 1:
                    result = Vector((self.shape[0], other.shape[1]))
                else:
                    result = Matrix((self.shape[0], other.shape[1]))
                result = [[sum(self.data[self_row][neutral] * other.data[neutral][other_col]
                               for neutral in range(self.shape[1]))
                           for other_col in range(other.shape[1])]
                          for self_row in range(self.shape[0])]
                return result
            raise TypeError("Type should be a Matrix/Vector/Scalar")
        except Exception as exc:
            print(f"Uncaught Exception: {exc.__class__.__name__} "
                  f"at line {exc.__traceback__.tb_lineno} "
                  f"of {__file__} -"
                  f"{exc}")
            return None

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __str__(self: Self):
        return self.__repr__()

    def __repr__(self: Self):
        return f"Values: {str(self.data)} - Shape: {str(self.shape)}"

    def T(self):
        pass


class Vector(Matrix):
    """ Vector manipulation class"""

    def __init__(self, arg):
        super().__init__(arg)
        if self.shape[0] != 1 and self.shape[1] != 1:
            raise ValueError("Not a valid vector shape.")

    def dot(self, other: Self):
        pass


if __name__ == "__main__":
    m_list = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    m_listw = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    m_tuple = (3, 2)
    matrix1 = Matrix(m_list)
    print(matrix1)
    matrix2 = Matrix(m_listw)
    print(matrix2)
    print("---------------------")
    print(matrix1 - matrix2)
    v_list = [[1., 2., 3.]]
    v_tuple = (3, 1)
    vector = Vector(v_list)
    print(vector)
    vector = Vector(v_tuple)
    print(vector)
