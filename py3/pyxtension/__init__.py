__author__ = 'andrei.suiu@gmail.com'

from typing import Type, Any


class ValidateError(ValueError):
    def __init__(self, args):
        ValueError.__init__(self, args)


def validate(expr, msg="Invalid argument", exc: 'Type[Exception]' = ValidateError):
    """
    If the expression val does not evaluate to True, then raise a ValidationError with msg
    """
    if not expr:
        raise exc(msg)

class PydanticValidated:
    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validator

    @classmethod
    def _pydantic_validator(cls, v):
        if not isinstance(v, cls):
            raise TypeError(f'{repr(v)} is of type {type(v)} but is expected to be of {cls}')
        return v


class frozendict(dict):
    __slots__ = (
        "_hash",
    )

    def __new__(cls: Type['frozendict'], *args: Any, **kwargs: Any) -> 'frozendict':
        new = super().__new__(cls, *args, **kwargs)
        new._hash = None
        return new

    def __hash__(self, *args, **kwargs):
        """Calculate the hash if all values are hashable, otherwise raises a TypeError."""

        if self._hash is not None:
            _hash = self._hash
        else:
            try:
                fs = frozenset(self.items())
                _hash = hash(fs)
            except TypeError:
                raise TypeError("Dictionary values are not hashable")
            self._hash = _hash

        return _hash

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    def copy(self) -> 'frozendict':
        """ Return the object itself, as it's immutable. """
        return self

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable
