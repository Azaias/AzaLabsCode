"""On-disk cache of model capability metadata (R-P-7).

The requirement's teeth are in its last clause: *missing metadata degrades to
"unknown," never blocks a call*. So every path here is best-effort. A corrupt cache
file, an unwritable directory, a `/models` endpoint that 500s -- all of them end in
`None` and a call that proceeds anyway. The only thing a metadata failure may cost
is a nicer context-length warning.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from azalabscode.providers.base import ModelInfo, ModelPricing

DEFAULT_TTL_S = 24 * 60 * 60
"""One day. Model catalogues change on the order of weeks; pricing rarely."""


def default_cache_dir() -> Path:
    """Where the cache lives, honouring the usual per-OS conventions."""

    override = os.environ.get("AZALABSCODE_CACHE_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "azalabscode" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base_path = Path(xdg) if xdg else Path.home() / ".cache"
    return base_path / "azalabscode"


class ModelsCache:
    """A TTL'd JSON file mapping model id to `ModelInfo`."""

    def __init__(
        self,
        *,
        provider: str,
        cache_dir: Path | None = None,
        ttl_s: float = DEFAULT_TTL_S,
    ) -> None:
        self.provider = provider
        self.ttl_s = ttl_s
        self.dir = cache_dir if cache_dir is not None else default_cache_dir()
        self.path = self.dir / f"models-{provider}.json"
        self._memory: dict[str, ModelInfo] | None = None
        self._loaded_at: float | None = None

    @property
    def fresh(self) -> bool:
        """True when the in-memory copy is within its TTL."""

        return self._loaded_at is not None and (time.time() - self._loaded_at) < self.ttl_s

    def load(self) -> dict[str, ModelInfo] | None:
        """Read the cache, or return `None` if absent, stale or unreadable."""

        if self._memory is not None and self.fresh:
            return self._memory
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        fetched_at = raw.get("fetched_at")
        if not isinstance(fetched_at, int | float):
            return None
        if (time.time() - fetched_at) >= self.ttl_s:
            return None
        try:
            models = {
                model_id: ModelInfo.model_validate(payload)
                for model_id, payload in raw.get("models", {}).items()
            }
        except Exception:
            return None
        self._memory = models
        self._loaded_at = fetched_at
        return models

    def store(self, models: dict[str, ModelInfo]) -> None:
        """Write the cache atomically. Failure is swallowed: this is a cache."""

        self._memory = models
        self._loaded_at = time.time()
        payload = {
            "provider": self.provider,
            "fetched_at": self._loaded_at,
            "models": {k: v.model_dump(mode="json") for k, v in models.items()},
        }
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            # Temp file in the destination directory: os.replace is only atomic
            # within a volume, and the system temp dir is often a different one.
            fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".models-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError:
            pass

    def get(self, model_id: str) -> ModelInfo | None:
        """One model's metadata from the cache, if present and fresh."""

        models = self.load()
        if models is None:
            return None
        return models.get(model_id)


def parse_openrouter_models(payload: dict[str, Any]) -> dict[str, ModelInfo]:
    """Convert a `/models` response body into `ModelInfo` records.

    Tolerant by design: a field that is missing, null, or a string where a number
    was expected yields `None` for that field rather than discarding the model. The
    catalogue is other people's data.
    """

    def as_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def as_int(value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    fetched_at = time.time()
    out: dict[str, ModelInfo] = {}
    for entry in payload.get("data", []):
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        pricing_raw = entry.get("pricing") or {}
        architecture = entry.get("architecture") or {}
        top_provider = entry.get("top_provider") or {}
        parameters = entry.get("supported_parameters") or []
        modalities = architecture.get("input_modalities") or []
        out[model_id] = ModelInfo(
            id=model_id,
            name=entry.get("name") if isinstance(entry.get("name"), str) else None,
            context_length=as_int(entry.get("context_length")),
            max_output_tokens=as_int(top_provider.get("max_completion_tokens")),
            supports_tools="tools" in parameters if parameters else None,
            supports_reasoning="reasoning" in parameters if parameters else None,
            supports_images="image" in modalities if modalities else None,
            pricing=ModelPricing(
                prompt=as_float(pricing_raw.get("prompt")),
                completion=as_float(pricing_raw.get("completion")),
                image=as_float(pricing_raw.get("image")),
                request=as_float(pricing_raw.get("request")),
            ),
            fetched_at=fetched_at,
        )
    return out


__all__ = [
    "DEFAULT_TTL_S",
    "ModelsCache",
    "default_cache_dir",
    "parse_openrouter_models",
]
