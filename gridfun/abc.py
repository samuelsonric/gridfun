from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Number


def norm(a, /):
    return abs(a) @ 1


def normalize(a, b, /):
    return b // (a @ abs(b))


class SignedMeasure(ABC):
    @abstractmethod
    def __abs__(self):
        ...

    @abstractmethod
    def __matmul__(self, other):
        ...

    def normalize(self, other):
        return normalize(self, other)


class MeasurableFunction(Callable, SignedMeasure):
    @abstractmethod
    def __floordiv__(self, other):
        ...
