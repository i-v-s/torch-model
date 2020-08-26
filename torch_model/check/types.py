from typing import NewType, TypeVar

Natural = NewType('Natural', int)


class SizeVar:
    def __init__(self, name: str):
        self.__name__ = name

    def __repr__(self):
        return self.__name__


def check_shape(shape: tuple):
    assert isinstance(shape, tuple)
    for s in shape:
        assert isinstance(s, int) or isinstance(s, SizeVar)
