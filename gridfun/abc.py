from abc import ABC, abstractmethod


def normalize(sm, fun):
    return fun // (sm @ fun)


class SignedMeasure(ABC):
    @abstractmethod
    def tensor_prod(self, other):
        ...

    @abstractmethod
    def __matmul__(self, other):
        ...

    def normalize(self, other):
        return normalize(self, other)
