# Author: ASU --<andrei.suiu@gmail.com>
import json
from abc import ABC
from dataclasses import MISSING, Field, asdict, field, fields
from typing import Any, Callable, Dict, Optional, Type, cast

from json_composite_encoder import JSONCompositeEncoder

try:
    from pydantic.v1 import BaseModel, Extra
except ImportError:
    from pydantic import BaseModel, Extra


class ExtModel(BaseModel):
    """
    Extended Model with custom JSON encoder.
    Extends the standard Pydantic model functionality by allowing arbitrary types and providing custom encoding.

    The main purpose of existing this class is to fix encoding issue in PyDantic and provide a properly working custom JSON encoder for arbitrary types.
    Ex: if you have a custom type like TS, which derives from float, PyDantic will ignore the custom encoder and use the default one for float.
    """

    def json(
        self,
        *,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        encoder: Optional[Callable[[Any], Any]] = None,
        **dumps_kwargs: Any,
    ) -> str:
        """
        Note: we don't override the dict() method since it doesn't need custom encoder, so only here we can apply custom encoders;

        Generate a JSON representation of the model, `include` and `exclude` arguments as per `dict()`.
        `encoder` is an optional function to supply as `default` to json.dumps(), other arguments as per `json.dumps()`.
        """
        if skip_defaults is not None:
            exclude_unset = skip_defaults
        encoder = cast(Callable[[Any], Any], encoder or self.__json_encoder__)
        data = self.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if self.__custom_root_type__:
            # below is a hardcoding workaround instead of original utils.ROOT_KEY as Pydantic doesn't have it on Unix
            data = data["__root__"]
        composite_encoder_builder = JSONCompositeEncoder.Builder(encoders=self.__config__.json_encoders)
        # Note: using a default arg instead of cls would not call encoder for elements that derive from base types like str or float;
        return self.__config__.json_dumps(data, default=encoder, cls=composite_encoder_builder, **dumps_kwargs)

    def replace(self, **changes: Any) -> "ExtModel":
        """
        Return a deep copy of the model replacing the specified fields with the supplied values.
        """
        return self.copy(update=changes, deep=True)

    class Config:
        arbitrary_types_allowed = True


class ImmutableExtModel(ExtModel):
    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True
        extra = Extra.forbid


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


COERCE_FIELD_NAME = "coerce"


def _coerce_type(type_: Type[Any], value: Any) -> Any:
    if isinstance(value, type_):
        return value
    try:
        return type_(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot coerce {value!r} to {type_.__name__}") from exc


# pylint: disable=redefined-builtin
def coercing_field(*, default=MISSING, default_factory=MISSING, init=True, repr=True, hash=None, compare=True, metadata=None) -> Field:
    metadata = field_config(metadata, coerce=True)
    # pylint: disable=invalid-field-call
    return field(default=default, default_factory=default_factory, init=init, repr=repr, hash=hash, compare=compare, metadata=metadata)


# pylint: disable=unused-argument
def field_config(
    metadata: Optional[dict] = None,
    *,
    coerce: bool = False,
    encoder: Optional[Callable] = None,
    decoder: Optional[Callable] = None,
    field_name: Optional[str] = None,
) -> Optional[Dict[str, dict]]:
    if coerce:
        if metadata is None:
            metadata = {}
        metadata[COERCE_FIELD_NAME] = True

    return metadata


class JsonData(ABC):
    """
    Intended to be used with @dataclass decorator.
    Dataclass with custom JSON encoder.
    Performs automatic type coercion and provides custom encoding.
    """

    class Config:
        json_encoders = {}

    def __post_init__(self):
        for field_ in fields(self):
            if not field_.metadata.get(COERCE_FIELD_NAME, False):
                continue
            value = getattr(self, field_.name)
            coerced_value = _coerce_type(field_.type, value)
            # Bypass the frozen nature of dataclasses to set the coerced value
            object.__setattr__(self, field_.name, coerced_value)

    def json(self, **dumps_kwargs: Any):
        data = asdict(self)
        combined_encoders = self._get_combined_encoders()
        composite_encoder_builder = JSONCompositeEncoder.Builder(encoders=combined_encoders)
        return json.dumps(data, cls=composite_encoder_builder, **dumps_kwargs)

    @classmethod
    def _get_combined_encoders(cls):
        combined_encoders = {}
        for base in reversed(cls.__mro__):
            if hasattr(base, "Config") and hasattr(base.Config, "json_encoders"):
                combined_encoders.update(base.Config.json_encoders)
        return combined_encoders
