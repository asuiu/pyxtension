# Author: ASU --<andrei.suiu@gmail.com>
from typing import Any

from pydantic import BaseModel


class ExtModel(BaseModel):
    """
    Extended Model with custom JSON encoder.
    Extends the standard Pydantic model functionality by allowing arbitrary types and providing custom encoding.

    The main purpose of existing this class is to fix encoding issues in Pydantic and provide a properly working custom JSON encoder for arbitrary types.
    Ex: if you have a custom type like TS, which derives from float, Pydantic will ignore the custom encoder and use the default one for float.
    """

    def replace(self, **changes: Any) -> "ExtModel":
        """
        Return a deep copy of the model replacing the specified fields with the supplied values.
        """
        return self.model_copy(update=changes, deep=True)

    model_config = {"arbitrary_types_allowed": True}


class ImmutableExtModel(ExtModel):
    model_config = {"frozen": True, "arbitrary_types_allowed": True, "extra": "forbid"}
