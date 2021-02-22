from collections.abc import Callable, Sequence
from numbers import Number

from gridfun.abc import SignedMeasure


def integrate(a, b, /):
    if isinstance(b, Number):
        return b
    if isinstance(b, Callable):
        return b(*a.point)

    return NotImplemented


def tensor_prod(a, b, /):
    if isinstance(b, Dirac):
        return Dirac(
            *a.point,
            *b.point,
        )

    return NotImplemented


def absolute(a, /):
    return a


class Dirac(SignedMeasure):
    def __init__(self, *point):
        self.point = point

    def __matmul__(self, other):
        return integrate(self, other)    

    def __abs__(self):
        return absolute(self)

    def tensor_prod(self, other):
        return tensor_prod(self, other)

    def __repr__(self):
        return f"{type(self).__name__}{self.point}"
