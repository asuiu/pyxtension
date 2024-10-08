from threading import RLock
from typing import Any, Dict, Iterable, MutableMapping, Optional, Tuple, TypeVar

__author__ = "andrei.suiu@gmail.com"

from streamerate import slist

_K = TypeVar("_K")
_V = TypeVar("_V")


class SafeDict(dict, Dict[_K, _V]):
    """
    Thread-safe dict.

    Note: keys(), values() and items() do not return the view objects as in standard dict, but returns the slist() objects with their copies.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._lock = RLock()

    def __getitem__(self, key) -> _V:
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        with self._lock:
            super().__delitem__(key)

    def __len__(self):
        with self._lock:
            return super().__len__()

    def keys(self) -> slist[_K]:
        with self._lock:
            return slist(super().keys())

    def values(self) -> slist[_V]:
        with self._lock:
            return slist(super().values())

    def items(self) -> slist[Tuple[_K, _V]]:
        with self._lock:
            return slist(super().items())

    def clear(self) -> None:
        with self._lock:
            super().clear()

    def copy(self) -> "SafeDict[_K, _V]":
        with self._lock:
            return self.__class__(super().copy())

    @classmethod
    def fromkeys(cls, __iterable: Iterable[_V], __value: Any = None) -> Dict[_V, Any]:
        return cls(dict.fromkeys(__iterable, __value))

    def __ior__(self, _value: Iterable[Tuple[_K, _V]]):
        with self._lock:
            try:
                return super().__ior__(_value)
            except AttributeError:
                for k, v in _value:
                    self[k] = v
                return self

    def popitem(self) -> Tuple[_K, _V]:
        with self._lock:
            return super().popitem()

    def setdefault(self: MutableMapping[_K, Optional[_V]], __key: _K) -> Optional[_V]:
        with self._lock:
            return super().setdefault(__key)

    def update(self, __m: Iterable[Tuple[_K, _V]], **kwargs: _V) -> None:
        with self._lock:
            super().update(__m, **kwargs)

    def __contains__(self, __o: object) -> bool:
        with self._lock:
            return super().__contains__(__o)

    def __eq__(self, __o: object) -> bool:
        with self._lock:
            return super().__eq__(__o)
