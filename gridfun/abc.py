from abc import ABC, abstractmethod


class SignedMeasure(ABC):
    @abstractmethod
    def tensor_prod(self, other):
        ...

    @abstractmethod
    def __matmul__(self, other):
        ...
