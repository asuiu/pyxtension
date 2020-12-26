__author__ = 'asuiu'

from typing import Type


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
