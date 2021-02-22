from collections.abc import Callable, Sequence
from itertools import product
from numbers import Number
from operator import methodcaller

import numpy as np

from gridfun.abc import SignedMeasure, MeasurableFunction
from gridfun.dirac import Dirac


def compose(a, b, /):
    if isinstance(a, (SignedMeasure, Number)):
        if isinstance(b, Cols):
            return np.fromiter(
                map(
                    a.__matmul__,
                    b,
                ),
                dtype="float",
                count=len(b),
            )
    if isinstance(a, Cols):
        if isinstance(b, Rows):
            return Kernel(a, b)
        if isinstance(b, np.ndarray):
            r = np.matmul(a, b)
            if b.ndim > 1:
                return Cols.from_iter(r)
            return r
    if isinstance(a, np.ndarray):
        if isinstance(b, Rows):
            r = np.matmul(a, b)
            if a.ndim > 1:
                return Rows.from_iter(r)
            return r
    if isinstance(a, Rows):
        if isinstance(b, (Callable, Number, Cols)):
            return np.array(
                tuple(
                    map(
                        methodcaller("__matmul__", b),
                        a,
                    )
                )
            )
    if isinstance(a, Kernel):
        return a.cols @ (a.rows @ b)
    if isinstance(b, Kernel):
        return (a @ b.cols) @ b.rows

    return NotImplemented


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
        return compose(self, other)

    def __rmatmul__(self, other):
        return compose(self, other)


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
        return compose(self, other)

    def __rmatmul__(self, other):
        return compose(self, other)


def iter_tensor_prod(a, b, /):
    for i, j in product(a, b):
        col = i[0] * j[0]
        row = i[1].tensor_prod(j[1])
        if not (col == 0 or row == 0):
            yield (col, row)


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
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows

    @classmethod
    def from_seqs(cls, cols, rows):
        return cls(
            Cols(cols),
            Rows(rows),
        )

    @classmethod
    def from_iters(cls, cols, rows):
        return cls.from_seqs(
            tuple(cols),
            tuple(rows),
        )

    @classmethod
    def from_measure(cls, sm):
        return cls.from_seqs(
            (1,),
            (sm,),
        )

    @classmethod
    def composition(cls, gf):
        u = np.unique(gf.y)
        return cls.from_iters(
            map(gf.preimg, u),
            map(Dirac, u),
        )

    @classmethod
    def conditional(cls, sm, ck):
        if not isinstance(ck, Kernel):
            ck = cls.composition(ck)
        return cls(
            ck.cols,
            Rows.from_iter(
                map(
                    sm.__mul__,
                    map(
                        sm.normalize,
                        ck.cols,
                    ),
                )
            ),
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, ix):
        return (
            self.cols[ix],
            self.rows[ix],
        )

    def __matmul__(self, other):
        return compose(self, other)

    def __rmatmul__(self, other):
        return compose(self, other)

    def __mul__(self, other):
        return tensor_prod(self, other)

    def __rmul__(self, other):
        return tensor_prod(self, other)
