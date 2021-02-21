from bisect import bisect
from collections.abc import Callable, Sequence
from functools import partial, reduce, cached_property
from heapq import merge
from itertools import starmap, repeat, groupby, compress, chain
from math import inf
from operator import attrgetter, itemgetter, methodcaller

import numpy as np


############ Endpoints ############


class Endpoint(Sequence):
    def __init__(self, x, cl):
        self.x = x
        self.cl = cl

    @classmethod
    def closed(cls, x):
        return cls(x, 1)

    def __len__(self):
        return 2

    def __getitem__(self, ix):
        return (self.x, self.cl)[ix]

    def __eq__(self, other):
        return self.x == other.x and self.cl == other.cl

    def __le__(self, other):
        return self.x < other.x or (self.cl and self.x == other.x)

    def __lt__(self, other):
        return self <= other and not self == other

    def __repr__(self):
        return f"{('(', '[')[self.cl]}{self.x:.2f}"


############ Axes ############


class Axis(Callable, Sequence):
    def __init__(self, eps):
        self.eps = eps

    @classmethod
    def from_iter(cls, it):
        return cls(tuple(it))

    @classmethod
    def halfopen(cls, x):
        return cls(
            (
                Endpoint(-inf, 0),
                *map(Endpoint.closed, x),
            )
        )

    @classmethod
    def empty(cls):
        return cls((Endpoint(-inf, 0),))

    @cached_property
    def leb(self):
        return np.fromiter(
            iter_leb(self),
            dtype="float",
            count=len(self),
        )

    @cached_property
    def midpoints(self):
        return np.fromiter(
            iter_midpoints(self),
            dtype="float",
            count=len(self),
        )

    def compress(self, selectors):
        return self.from_iter(
            compress(
                self,
                selectors,
            )
        )

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, ix):
        return self.eps[ix]

    def __call__(self, x):
        return bisect(self, Endpoint(x, 1)) - 1

    def __eq__(self, other):
        return self.eps == other.eps

    def __str__(self):
        return "[" + " ".join(map(str, self)) + "]"

    def __repr__(self):
        return f"{type(self).__name__}([" + ", ".join(map(str, self)) + "])"


def index_axes(axes):
    for ax, ix in zip(axes, np.eye(len(axes), dtype="int")):
        yield zip(ax, repeat(ix))


def join_axes(axes):
    eps, sel = zip(
        *starmap(
            lambda a, b: (a, sum(map(itemgetter(1), b))),
            groupby(
                merge(
                    *index_axes(axes),
                    key=itemgetter(0),
                ),
                key=itemgetter(0),
            ),
        )
    )

    return (Axis(eps), np.array(sel).T)


def iter_leb(ax):
    i = ax[0]
    for j in ax[1:]:
        yield (j.x - i.x)
        i = j
    yield inf


def iter_midpoints(ax):
    i = ax[0]
    for j in ax[1:]:
        yield 0.5 * (i.x + j.x)
        i = j
    yield inf


############ Grids ############


class Grid(Callable, Sequence):
    def __init__(self, axes):
        self.axes = axes

    @classmethod
    def from_iter(cls, it):
        return cls(tuple(it))

    @classmethod
    def from_axis(cls, axis):
        return cls((axis,))

    @classmethod
    def linspace(cls, start, stop, num=50):
        arr = np.linspace(start, stop, num).T
        if arr.ndim > 1:
            return cls.from_iter(
                map(
                    Axis.halfopen,
                    arr,
                )
            )

        return cls.from_axis(Axis.halfopen(arr))

    @classmethod
    def empty(cls, ndim):
        return cls.from_iter(repeat(Axis.empty(), ndim))

    @cached_property
    def shape(self):
        return tuple(map(len, self))

    @cached_property
    def leb(self):
        with np.errstate(invalid="ignore"):
            arr = reduce(
                partial(np.tensordot, axes=0),
                map(
                    attrgetter("leb"),
                    self,
                ),
            )
        arr[np.isnan(arr)] = 0
        return arr

    @cached_property
    def midpoints(self):
        return np.stack(
            np.meshgrid(
                *map(attrgetter("midpoints"), self),
                indexing="ij",
            ),
            axis=-1,
        )

    def eval(self, fun):
        return np.apply_along_axis(
            lambda x: abs(1.0 - np.isinf(x).any()) and fun(*x),
            arr=self.midpoints,
            axis=-1,
        )

    def compress(self, selectors, axis):
        return type(self)(
            (
                *self[:axis],
                self[axis].compress(selectors),
                *self[axis + 1 :],
            )
        )

    def __len__(self):
        return len(self.axes)

    def __getitem__(self, ix):
        return self.axes[ix]

    def __eq__(self, other):
        return self.axes == other.axes

    def __call__(self, *x):
        return tuple(ax(i) for ax, i in zip(self, x))

    def __str__(self):
        return "\n".join(map(str, self))

    def __repr__(self):
        return f"{type(self).__name__}(" + "\n     ".join(map(str, self)) + ")"


def join_grids(grids):
    axes, sel = zip(
        *map(
            join_axes,
            zip(*grids),
        )
    )

    return (Grid(axes), sel)
