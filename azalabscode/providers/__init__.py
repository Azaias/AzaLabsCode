"""Model access.

Public API of the provider layer. `Provider` is a protocol with one required
method, `stream()`; `complete()` is a helper over it rather than a second path
(R-P-1). Nothing in `ModelRequest` or `StreamEvent` names a vendor -- provider
knobs travel in `provider_options`, opaque to every other layer (R-P-2).

    provider = OpenRouterProvider()
    async for event in provider.stream(ModelRequest(model="...", messages=[...])):
        ...
"""

from azalabscode.providers.base import (
    Finish,
    ModelInfo,
    ModelPricing,
    ModelRequest,
    Provider,
    ReasoningConfig,
    ReasoningDelta,
    StreamAccumulator,
    StreamError,
    StreamEvent,
    StreamEventUnion,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageReport,
    complete,
)
from azalabscode.providers.models_cache import ModelsCache
from azalabscode.providers.openrouter import OpenRouterProvider
from azalabscode.providers.testing import (
    FakeProvider,
    Script,
    ScriptedToolCall,
    ScriptedTurn,
    tool_call,
    turn,
)

__all__ = [
    "FakeProvider",
    "Finish",
    "ModelInfo",
    "ModelPricing",
    "ModelRequest",
    "ModelsCache",
    "OpenRouterProvider",
    "Provider",
    "ReasoningConfig",
    "ReasoningDelta",
    "Script",
    "ScriptedToolCall",
    "ScriptedTurn",
    "StreamAccumulator",
    "StreamError",
    "StreamEvent",
    "StreamEventUnion",
    "TextDelta",
    "ToolCallDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "UsageReport",
    "complete",
    "tool_call",
    "turn",
]
