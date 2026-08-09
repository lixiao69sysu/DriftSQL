"""Small VERL tool-contract fallback for the standalone API/CLI runtime.

Training uses the pinned VERL implementation whenever it is installed. The
product service only needs the tool schema, response object, base constructor,
and tracing decorator, so a source checkout of the full training framework is
not required in a deployment image.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

_FORCE_PORTABLE = os.getenv("DRIFTSQL_PORTABLE_TOOL_RUNTIME", "").strip().casefold() in {
    "1",
    "true",
    "yes",
    "on",
}

USING_VERL = False

if not _FORCE_PORTABLE:
    try:
        from verl.tools.base_tool import BaseTool as BaseTool
        from verl.tools.schemas import OpenAIFunctionToolSchema as OpenAIFunctionToolSchema
        from verl.tools.schemas import ToolResponse as ToolResponse
        from verl.utils.rollout_trace import rollout_trace_op as rollout_trace_op

        USING_VERL = True
    except ModuleNotFoundError as error:
        # Fall back only when VERL itself is absent. A partially installed VERL
        # must fail loudly instead of silently changing the training runtime.
        if error.name != "verl" and not str(error.name).startswith("verl."):
            raise


if not USING_VERL:

    class OpenAIFunctionPropertySchema(BaseModel):
        type: str | list[str]
        description: str | None = None
        enum: list[Any] | None = None


    class OpenAIFunctionParametersSchema(BaseModel):
        type: str = "object"
        properties: dict[str, OpenAIFunctionPropertySchema] = Field(default_factory=dict)
        required: list[str] = Field(default_factory=list)


    class OpenAIFunctionSchema(BaseModel):
        name: str
        description: str
        parameters: OpenAIFunctionParametersSchema = Field(default_factory=OpenAIFunctionParametersSchema)
        strict: bool = False


    class OpenAIFunctionToolSchema(BaseModel):
        type: Literal["function"] = "function"
        function: OpenAIFunctionSchema


    class ToolResponse(BaseModel):
        text: str | None = None
        image: list[Any] | None = None
        video: list[Any] | None = None

        @model_validator(mode="before")
        @classmethod
        def validate_media_lists(cls, values: Any) -> Any:
            if isinstance(values, dict):
                for key in ("image", "video"):
                    if key in values and values[key] is not None and not isinstance(values[key], list):
                        raise ValueError(f"{key} must be a list")
            return values

        def is_empty(self) -> bool:
            return not self.text and not self.image and not self.video

        def is_text_only(self) -> bool:
            return bool(self.text and not self.image and not self.video)


    class BaseTool:
        """Portable subset of VERL's BaseTool used by DriftSQL tools."""

        def __init__(self, config: dict[str, Any], tool_schema: OpenAIFunctionToolSchema):
            self.config = config
            self.tool_schema = tool_schema
            self.name = tool_schema.function.name
            print(json.dumps(tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

        def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
            return self.tool_schema

        async def create(self, instance_id: str | None = None, **kwargs: Any) -> tuple[str, ToolResponse]:
            del kwargs
            return instance_id or str(uuid4()), ToolResponse()

        async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
            del instance_id, kwargs
            return 0.0

        async def release(self, instance_id: str, **kwargs: Any) -> None:
            del instance_id, kwargs


    _Function = TypeVar("_Function", bound=Callable[..., Any])

    def rollout_trace_op(function: _Function) -> _Function:
        """No-op replacement for VERL rollout tracing in serving processes."""

        return function


__all__ = [
    "BaseTool",
    "OpenAIFunctionToolSchema",
    "ToolResponse",
    "USING_VERL",
    "rollout_trace_op",
]
