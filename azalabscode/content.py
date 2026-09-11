"""Content parts: the atoms a message or a tool result is made of.

Split out of `messages` (spec delta 1) because `ToolResult.content` is a list of
these and `ToolResultMessage.result` is a `ToolResult`. Keeping all three in one
module, as spec 4.2 does, creates a `messages` <-> `tools` import cycle that the
spec's own layering contract forbids.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import Field

from azalabscode.schema import HarnessModel


class TextPart(HarnessModel):
    """Plain text produced by a model, a user, or a tool."""

    type: Literal["text"] = "text"
    text: str


class ReasoningPart(HarnessModel):
    """Model reasoning / thinking content, where the model emits it.

    `signature` carries whatever opaque token the provider requires to round-trip
    the block on a later turn. It is never interpreted here.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None
    redacted: bool = False
    """True when the provider returned an encrypted placeholder rather than text."""


class ToolCallPart(HarnessModel):
    """A model's request to call a tool.

    `raw_arguments` is always the exact string the model produced. `arguments` is
    the parsed form, or `None` with `parse_error` set -- malformed JSON is data, not
    an exception (R-P-4). The agent loop turns a `parse_error` into a structured
    tool error so the model gets a chance to fix its own output.
    """

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str = ""
    parse_error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the arguments parsed and the call can be dispatched."""

        return self.parse_error is None and self.arguments is not None


class ImagePart(HarnessModel):
    """An image, base64-encoded. Produced by `read_file` on image files."""

    type: Literal["image"] = "image"
    media_type: str
    data_b64: str
    detail: Literal["auto", "low", "high"] = "auto"


class FilePart(HarnessModel):
    """Reserved for non-image binary attachments. Not populated in v1."""

    type: Literal["file"] = "file"
    media_type: str
    filename: str | None = None
    data_b64: str | None = None
    uri: str | None = None


Part = Annotated[
    TextPart | ReasoningPart | ToolCallPart | ImagePart | FilePart,
    Field(discriminator="type"),
]
"""Discriminated union of every content part. Use this in model annotations."""

ContentPart = Part
"""Alias used where "part of a tool result" reads better than "part of a message"."""


def parse_tool_arguments(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool-call argument string into `(arguments, parse_error)`.

    Never raises. An empty string means "no arguments" and parses to `{}`, because
    several models emit `""` for a zero-parameter tool. A JSON value that is not an
    object is an error: tool parameters are always a mapping.
    """

    stripped = raw.strip()
    if not stripped:
        return {}, None
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, f"expected a JSON object, got {type(value).__name__}"
    return value, None


def text_of(parts: list[Part]) -> str:
    """Concatenate the text of every `TextPart`, ignoring everything else.

    The convenience accessor behind `AssistantMessage.text` and `AgentResult.
    final_text`. Reasoning is deliberately excluded: it is not the answer.
    """

    return "".join(p.text for p in parts if isinstance(p, TextPart))


def tool_calls_of(parts: list[Part]) -> list[ToolCallPart]:
    """Every `ToolCallPart` in `parts`, in order."""

    return [p for p in parts if isinstance(p, ToolCallPart)]


__all__ = [
    "ContentPart",
    "FilePart",
    "ImagePart",
    "Part",
    "ReasoningPart",
    "TextPart",
    "ToolCallPart",
    "parse_tool_arguments",
    "text_of",
    "tool_calls_of",
]
