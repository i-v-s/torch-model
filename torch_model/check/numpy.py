import numpy as np
from .types import check_shape


class NDArrayItem:
    def __init__(self, *params):
        self.__origin__ = np.ndarray
        params = params if isinstance(params, tuple) else (params,)
        self.shape = params[:-1]
        check_shape(self.shape)
        self.dtype = params[-1]

    def sum_type(self, other: 'NDArrayItem'):
        result = (np.zeros((), self.dtype) + np.zeros((), other.dtype)).dtype
        return result.type

    def __add__(self, other: 'NDArrayItem'):
        assert self.shape == other.shape
        return NDArrayItem(*self.shape, self.sum_type(other))

    def __mul__(self, other: 'NDArrayItem'):
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        return NDArrayItem(*self.shape, self.dtype)

    def __isub__(self, other):
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        return NDArrayItem(*self.shape, self.dtype)

    def __repr__(self):
        params = ','.join(map(str, self.shape + (self.dtype,)))
        return f'NDArray[{params}]'

    def __eq__(self, other):
        if not isinstance(other, NDArrayItem):
            return False
        return self.dtype, self.shape == other.dtype, other.shape

    def __call__(self):
        return np.zeros(self.shape, self.dtype)


class _NDArray:
    def __init__(self):
        self.__origin__ = np.ndarray

    def __getitem__(self, item):
        return NDArrayItem(*item)


class NumPy:
    uint8 = np.uint8
    @staticmethod
    def zeros(shape, dtype=None, order='C'):
        if dtype is None:
            dtype = np.float64
        return NDArrayItem(*shape, dtype)


NDArray = _NDArray()
