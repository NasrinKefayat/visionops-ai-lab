"""Tool contracts only; no agent loop or external calls are implemented."""

from typing import Protocol

from pydantic import BaseModel, Field


class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ToolResult(BaseModel):
    content: str
    success: bool


class AssistantTool(Protocol):
    name: str

    def execute(self, request: ToolRequest) -> ToolResult:
        """Implementations must validate arguments and enforce their own access policy."""
        ...
