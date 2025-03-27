__author__ = "andrei.suiu@gmail.com"

from typing import Any, Type

try:
    from pydantic.v1 import ValidationError
except ImportError:
    from pydantic import ValidationError

from pyxtension.safe_dict import SafeDict  # noqa: F401

try:
    from pydantic_core.core_schema import any_schema as pydantic_any_schema
    from pydantic_core.core_schema import list_schema as pydantic_list_schema
    from pydantic_core.core_schema import no_info_after_validator_function as pydantic_no_info_after_validator_function
    from pydantic_core.core_schema import no_info_plain_validator_function as pydantic_no_info_plain_validator_function
except ImportError:

    def __func_raising(*args, **kwargs):
        raise ImportError("pydantic V2 is not installed")

    pydantic_list_schema = __func_raising
    pydantic_any_schema = __func_raising
    pydantic_no_info_after_validator_function = __func_raising
    pydantic_no_info_plain_validator_function = __func_raising


class ValidateError(ValueError):
    def __init__(self, *args, **kwargs):
        ValueError.__init__(self, *args, **kwargs)


def validate(expr, msg="Invalid argument", exc: "Type[Exception]" = ValidateError):
    """
    If the expression val does not evaluate to True, then raise a ValidationError with msg
    """
    if not expr:
        raise exc(msg)


class PydanticStrictValidated:
    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validator

    @classmethod
    def _pydantic_validator(cls, v):
        if not isinstance(v, cls):
            raise ValidationError(f"{repr(v)} is of type {type(v)} but is expected to be of {cls}")
        return v

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any):
        """
        This will work only with plain types. If you need collection/lists, then use next:
            `pydantic_no_info_after_validator_function(cls, pydantic_list_schema(items_schema=pydantic_any_schema()))`
        """
        return pydantic_no_info_plain_validator_function(cls)


# PydanticValidated is deprecated. Use one of the Pydantic<Coercing|Strict>Validated classes
PydanticValidated = PydanticStrictValidated


class PydanticCoercingValidated:
    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validator

    @classmethod
    def _pydantic_validator(cls, v):
        if not isinstance(v, cls):
            return cls(v)
        return v

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any):
        """
        This will work only with plain types. If you need collection/lists, then use next:
            `pydantic_no_info_after_validator_function(cls, pydantic_list_schema(items_schema=pydantic_any_schema()))`
        """
        return pydantic_no_info_plain_validator_function(cls)


class frozendict(dict):
    __slots__ = ("_hash",)

    def __new__(cls: Type["frozendict"], *args: Any, **kwargs: Any) -> "frozendict":
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
            except TypeError as exc:
                raise TypeError("Dictionary values are not hashable") from exc
            # pylint: disable=attribute-defined-outside-init
            self._hash = _hash

        return _hash

    # pylint: disable=no-self-use
    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    def copy(self) -> "frozendict":
        """Return the object itself, as it's immutable."""
        return self

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable
