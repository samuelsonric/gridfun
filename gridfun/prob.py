from collections.abc import Callable, Sequence
from itertools import product
from numbers import Number
from operator import methodcaller

import numpy as np

from gridfun.abc import SignedMeasure


############ Dirac Measures ############


class Dirac(SignedMeasure):
    def __init__(self, *point):
        self.point = point

    def __matmul__(self, other):
        if isinstance(other, Callable):
            return other(*self.point)

        return NotImplemented

    def tensor_prod(self, other):
        if isinstance(other, Dirac):
            return Dirac(
                *self.point,
                *other.point,
            )

        return NotImplemented

    def __repr__(self):
        return f"{type(self).__name__}{self.point}"


############ Rows and Cols ############


class Rows(Sequence):
    def __init__(self, rows):
        self.rows = rows

    @classmethod
    def from_iter(cls, rows):
        return cls(tuple(rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        return self.rows[ix]

    def __matmul__(self, other):
        if isinstance(other, Kernel):
            return self.from_iter(
                map(
                    methodcaller("__matmul__", other),
                    self,
                )
            )
        if isinstance(other, (Callable, Number)):
            return np.fromiter(
                map(
                    methodcaller("__matmul__", other),
                    self,
                ),
                dtype="float",
                count=len(self),
            )
        if isinstance(other, Cols):
            return np.array(
                tuple(
                    map(
                        methodcaller("__matmul__", other),
                        self,
                    )
                )
            )

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray):
            arr = np.matmul(
                other,
                self,
            )
            if arr.size > 1:
                return self.from_iter(arr)

            return arr.item()

        return NotImplemented


class Cols(Sequence):
    def __init__(self, cols):
        self.cols = cols

    @classmethod
    def from_iter(cls, cols):
        return cls(tuple(cols))

    def __len__(self):
        return len(self.cols)

    def __getitem__(self, ix):
        return self.cols[ix]

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            r = np.matmul(
                self,
                other,
            )
            if other.ndim > 1:
                return self.from_iter(ar)

            return r

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, Kernel):
            return self.from_iter(
                map(
                    other.__matmul__,
                    self,
                )
            )
        if isinstance(other, (SignedMeasure, Number)):
            return np.fromiter(
                map(other.__matmul__, self),
                dtype="float",
                count=len(self),
            )

        return NotImplemented


############ Kernels ############


def iter_tensor_prod(a, b, /):
    for i, j in product(a, b):
        row = i[0].tensor_prod(j[0])
        col = i[1] * j[1]
        if not (row == 0 or col == 0):
            yield (row, col)


def tensor_prod(a, b, /):
    if isinstance(a, SignedMeasure):
        return tensor_prod(
            Kernel.from_measure(a),
            b,
        )
    if isinstance(b, SignedMeasure):
        return tensor_prod(
            a,
            Kernel.from_measure(b),
        )
    if isinstance(a, Kernel) and isinstance(b, Kernel):
        return Kernel.from_seqs(
            *zip(
                *iter_tensor_prod(
                    a,
                    b,
                )
            )
        )

    return NotImplemented


class Kernel(Sequence):
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    @classmethod
    def from_seqs(cls, rows, cols):
        return cls(
            Rows(rows),
            Cols(cols),
        )

    @classmethod
    def from_iters(cls, rows, cols):
        return cls.from_seqs(
            tuple(rows),
            tuple(cols),
        )

    @classmethod
    def from_measure(cls, sm):
        return cls.from_seqs(
            (sm,),
            (1,),
        )

    @classmethod
    def composition(cls, gf):
        u = np.unique(gf.y)
        return cls.from_iters(
            map(Dirac, u),
            map(gf.preimg, u),
        )

    @classmethod
    def conditional(cls, sm, ck):
        if not isinstance(ck, Kernel):
            ck = cls.composition(ck)
        return cls(
            Rows.from_iter(
                map(
                    sm.__mul__,
                    map(
                        sm.normalize,
                        ck.cols,
                    ),
                )
            ),
            ck.cols,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        return (
            self.rows[ix],
            self.cols[ix],
        )

    def __matmul__(self, other):
        if isinstance(other, Kernel):
            return Kernel(
                self.cols,
                self.rows @ other,
            )
        if isinstance(other, (Callable, Number)):
            return self.cols @ (self.rows @ other)

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, (Callable, Number)):
            return (other @ self.cols) @ self.rows

        return NotImplemented

    def __mul__(self, other):
        return tensor_prod(self, other)

    def __rmul__(self, other):
        return tensor_prod(self, other)

    def __repr__(self):
        return f"{type(self).__name__}()"
