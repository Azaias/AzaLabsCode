"""Pydantic base classes shared by every model in the harness.

This is the bottom of the dependency graph: it imports nothing from the package.

`HarnessModel` fixes the configuration every core model needs -- extra fields are
an error rather than silently dropped, and enum values serialize as their value so
JSON stays readable and diffable. `VersionedModel` adds the `schema_version` field
required by R-X-4 for anything that is written to disk or crosses a process
boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1
"""Current version of the on-disk document formats.

Bump when a `VersionedModel` subclass changes shape in a way that an older reader
cannot handle. Readers compare against the value on the document, not this
constant, so an old file always reports the version it was written with.
"""


class HarnessModel(BaseModel):
    """Base for all harness data models.

    - `extra="forbid"`: an unknown key is a bug (a rename, a typo, a version skew),
      not something to swallow. Round-trip fidelity (R-X-4) depends on it.
    - `use_enum_values=False`: enums stay enums in Python and serialize by value.
    - `validate_assignment=True`: mutating a field re-validates, so a model that
      passed validation cannot be corrupted after the fact and then checkpointed.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class VersionedModel(HarnessModel):
    """A `HarnessModel` that is serialized to disk or across a process boundary."""

    schema_version: int = SCHEMA_VERSION


class OpenModel(BaseModel):
    """Base for models that intentionally carry provider- or tool-defined keys.

    Used for opaque bags (`provider_options`, `meta`) where forbidding extras
    would defeat the point.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)


def json_safe(value: Any) -> bool:
    """True when `value` survives a JSON round-trip unchanged.

    Used by the two-stage serialization check (spec C-3): stage one calls this on
    a node's default state at build time, long before a model call has cost money.
    """

    import json

    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


__all__ = [
    "SCHEMA_VERSION",
    "HarnessModel",
    "OpenModel",
    "VersionedModel",
    "json_safe",
]
