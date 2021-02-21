from collections.abc import Callable
from functools import cached_property, partial
from itertools import zip_longest, starmap, chain
from numbers import Number
from operator import attrgetter

import numpy as np
from wrapt import decorator

from gridfun.grid import Grid, join_grids
from gridfun.abc import SignedMeasure


############ Y ############


def iter_selectors(y, axis):
    i = None
    for j in np.rollaxis(y, axis):
        if (s := (i != j).any()) :
            i = j
        yield s


def selectors(y, axis):
    return tuple(iter_selectors(y, axis))


def uncompress(y, selectors, axis):
    return y.take(
        np.cumsum(selectors) - 1,
        axis,
    )


############ Binary Operations ############


def gf_gf(op, a, b, /, compress=True):
    gr, (x, y) = join_gfs((a, b))
    gf = GridFun(gr, op(x, y))
    if compress:
        gf = autocompress(gf)
    return gf


def gf_c(op, a, b, /, compress=True):
    gf = GridFun(a.grid, op(a.y, b))
    if compress:
        gf = autocompress(gf)
    return gf


def c_gf(op, a, b, /, compress=True):
    gf = GridFun(b.grid, op(a, b.y))
    if compress:
        gf = autocompress(gf)
    return gf


def binary_op(inj=False, sc=False):
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        a, b = args
        if isinstance(a, GridFun):
            if isinstance(b, GridFun):
                return gf_gf(
                    wrapped,
                    a,
                    b,
                    compress=True,
                )
            if isinstance(b, Number):
                if not b and sc:
                    return a.zero(a.ndim)
                else:
                    return gf_c(
                        wrapped,
                        a,
                        b,
                        compress=not inj,
                    )
        if isinstance(a, Number):
            if isinstance(b, GridFun):
                if not a and sc:
                    return b.zero(b.ndim)
                else:
                    return c_gf(
                        wrapped,
                        a,
                        b,
                        compress=not inj,
                    )
            if isinstance(b, Number):
                return wrapped(a, b)

        return NotImplemented

    return wrapper


@binary_op(inj=True, sc=False)
def add(a, b, /):
    return np.add(
        a,
        b,
    )


@binary_op(inj=True, sc=False)
def subtract(a, b, /):
    return np.subtract(
        a,
        b,
    )


@binary_op(inj=True, sc=True)
def multiply(a, b, /):
    return np.multiply(
        np.where(b != 0, a, 0),
        np.where(a != 0, b, 0),
    )


@binary_op(inj=True, sc=True)
def floor_divide(a, b, /):
    return np.divide(
        np.where(b != 0, a, 0),
        np.where(b != 0, b, 1),
    )


@binary_op(inj=False, sc=False)
def minimum(a, b, /):
    return np.minimum(
        a,
        b,
    )


@binary_op(inj=False, sc=False)
def maximum(a, b, /):
    return np.maximum(
        a,
        b,
    )


@binary_op(inj=False, sc=False)
def mod(a, b, /):
    return np.multiply(
        a,
        np.equal(a, b),
    )


def tensor_prod_gf_gf(a, b, /):
    if isinstance(a, GridFun) and isinstance(b, GridFun):
        return autocompress(
            GridFun(
                Grid.from_iter(chain(a.grid, b.grid)),
                np.tensordot(a.y, b.y, axes=0),
            )
        )


############ Scalar Operations ############


def scalar_op(inj=False):
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        a, b = args
        if isinstance(b, Number):
            return gf_c(
                wrapped,
                a,
                b,
                compress=not inj,
            )

        return NotImplemented

    return wrapper


@scalar_op(inj=False)
def preimg(a, b, /):
    return np.equal(
        a,
        b,
    )


@scalar_op(inj=False)
def clip(a, a_min, a_max, /):
    return np.clip(
        a,
        a_min,
        a_max,
    )


@scalar_op(inj=False)
def power(a, b, /):
    return np.power(
        a,
        b,
    )


############ Unary Operations ############


def _gf(op, a, inj=False):
    gf = GridFun(a.grid, op(a.y))
    if not inj:
        gf = autocompress(gf)
    return gf


def unary_op(inj=False):
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        a = args[0]
        return _gf(
            wrapped,
            a,
            compress=not inj,
        )

    return wrapper


@unary_op(inj=True)
def negative(a, /):
    return np.negative(
        a,
    )


@unary_op(inj=False)
def absolute(a, /):
    return np.absolute(
        a,
    )


############ Comparison ############


def equal(a, b, /):
    if isinstance(a, GridFun):
        if isinstance(b, GridFun):
            return a.grid == b.grid and (a.y == b.y).all()
        if isinstance(b, Number):
            return a.size == 1 and a.y.item() == b
    if isinstance(a, Number):
        if isinstance(b, GridFun):
            return b.size == 1 and b.y.item() == a
        if isinstance(b, Number):
            return a == b

    return NotImplemented


############ Grid Functions ############


def join_gfs(gfs):
    grids, ys = zip(
        *map(
            attrgetter("grid", "y"),
            gfs,
        )
    )
    gr, sel = join_grids(grids)
    for i, s in enumerate(sel):
        ys = starmap(
            partial(uncompress, axis=i),
            zip(ys, s),
        )

    return (gr, ys)


def autocompress(gf):
    for i in range(gf.ndim):
        gf = gf.compress(
            selectors(gf.y, i),
            i,
        )
    return gf


class GridFun(SignedMeasure, Callable):
    def __init__(self, grid, y):
        self.grid = grid
        self.y = y

    @classmethod
    def zero(cls, ndim):
        return cls(
            Grid.empty(ndim),
            np.zeros(np.ones(ndim, dtype="int")),
        )

    @classmethod
    def one(cls, ndim):
        return cls(
            Grid.empty(ndim),
            np.ones(np.ones(ndim, dtype="int")),
        )

    @classmethod
    def approx(cls, fun, start, stop, num=50):
        gr = Grid.linspace(start, stop, num)
        return cls(
            gr,
            gr.eval(fun),
        )

    @property
    def ndim(self):
        return self.y.ndim

    @property
    def shape(self):
        return self.y.shape

    @property
    def size(self):
        return self.y.size

    @cached_property
    def leb(self):
        arr = np.multiply(
            np.where(self.y != 0, self.grid.leb, 0),
            np.where(self.grid.leb != 0, self.y, 0),
        )
        return arr.sum()

    def compress(self, selectors, axis):
        return type(self)(
            self.grid.compress(selectors, axis),
            self.y.compress(selectors, axis),
        )

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __floordiv__(self, other):
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        return floor_divide(other, self)

    def __and__(self, other):
        return minimum(self, other)

    def __rand__(self, other):
        return minimum(other, self)

    def __or__(self, other):
        return maximum(self, other)

    def __ror__(self, other):
        return maximum(other, self)

    def __mod__(self, other):
        return mod(self, other)

    def __rmod__(self, other):
        return mod(other, self)

    def __matmul__(self, other):
        if isinstance(other, (GridFun, Number)):
            return (self * other).leb
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, (GridFun, Number)):
            return (other * self).leb
        return NotImplemented

    def tensor_prod(self, other):
        return tensor_prod_gf_gf(self, other)

    def __neg__(self):
        return negative(self)

    def __abs__(self):
        return absolute(self)

    def __call__(self, *x):
        return self.y[self.grid(*x)]

    def preimg(self, x):
        return preimg(self, x)

    def clip(self, a_min, a_max):
        return clip(self, a_min, a_max)

    def __pow__(self, x):
        return power(self, x)

    def __eq__(self, other):
        return equal(self, other)

    def __le__(self, other):
        return self == self & other

    def __lt__(self, other):
        return self <= other and not self == other

    def __repr__(self):
        return (
            f"{type(self).__name__}(grid="
            + "\n             ".join(str(self.grid).split("\n"))
            + "\n\n        y="
            + "\n          ".join(str(self.y).split("\n"))
            + ")"
        )
