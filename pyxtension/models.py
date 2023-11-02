# Author: ASU --<andrei.suiu@gmail.com>
import json
from dataclasses import dataclass, fields, asdict
from typing import Any, Callable, cast, Optional, Type

from json_composite_encoder import JSONCompositeEncoder

try:
    from pydantic.v1 import BaseModel, Extra
except ImportError:
    from pydantic import BaseModel, Extra


class ExtModel(BaseModel):
    """
    Extended Model with custom JSON encoder.
    Extends the standard Pydantic model functionality by allowing arbitrary types and providing custom encoding.
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


def _coerce_type(type_: Type[Any], value: Any) -> Any:
    if isinstance(value, type_):
        return value
    try:
        return type_(value)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot coerce {value!r} to {type_.__name__}")


class BaseSmartDataclass:
    """
    Smart Dataclass with custom JSON encoder.
    Performs automatic type coercion and provides custom encoding.
    """

    class Config:
        json_encoders = {}

    def __post_init__(self):
        for field_ in fields(self):
            value = getattr(self, field_.name)
            coerced_value = _coerce_type(field_.type, value)
            # Bypass the frozen nature of dataclasses to set the coerced value
            object.__setattr__(self, field_.name, coerced_value)

    def json(self, **dumps_kwargs: Any):
        data = asdict(self)
        combined_encoders = self._get_combined_encoders()
        composite_encoder_builder = JSONCompositeEncoder.Builder(encoders=combined_encoders)
        return json.dumps(data, cls=composite_encoder_builder, **dumps_kwargs)

    def _get_combined_encoders(self):
        combined_encoders = {}
        for base in reversed(self.__class__.__mro__):
            if hasattr(base, 'Config') and hasattr(base.Config, 'json_encoders'):
                combined_encoders.update(base.Config.json_encoders)
        return combined_encoders


@dataclass
class SmartDataclass(BaseSmartDataclass):
    """
    Smart Dataclass with custom JSON encoder.
    Performs automatic type coercion and provides custom encoding.
    """


@dataclass(frozen=True)
class FrozenSmartDataclass(BaseSmartDataclass):
    """
    Smart Immutable Dataclass with custom JSON encoder.
    Performs automatic type coercion and provides custom encoding.
    """
