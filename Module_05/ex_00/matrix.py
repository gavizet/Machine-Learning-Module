"""" Manipulation and understanding of basic matrix operations.
Numpy usage is forbidden"""


class Matrix:

    @staticmethod
    def __check_args(arg):
        pass

    @staticmethod
    def __get_values(arg):
        pass

    @staticmethod
    def __get_shape(arg):
        pass

    def __init__(self, arg):
        self.__check_args(arg)
        self.values = self.__get_values(arg)
        self.shape = self.__get_shape(arg)

    def __add__(self, other):
        pass

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def T(self):
        pass


class Vector(Matrix):

    def __init__(self):
        super().__init__()

    def dot(self, v: Vector):
        pass
