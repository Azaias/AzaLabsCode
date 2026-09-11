"""Stream a real completion from OpenRouter.

Half of M0's exit test: proof that the provider works against the live API and not
only against recorded fixtures.

    python scripts/stream_openrouter.py "Explain a ULID in one sentence."
    python scripts/stream_openrouter.py --model openai/gpt-4o-mini --tools "Read a.py"

The key comes from `OPENROUTER_API_KEY` or from a `.env` in the repo root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from azalabscode import (
    ModelRequest,
    OpenRouterProvider,
    ReasoningConfig,
    StreamAccumulator,
    SystemMessage,
    ToolSchema,
    UserMessage,
    run_sync,
)
from azalabscode.providers.base import (
    Finish,
    ReasoningDelta,
    StreamError,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageReport,
)

DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

READ_FILE_SCHEMA = ToolSchema(
    name="read_file",
    description="Read a file from the working directory and return its contents.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "limit": {"type": "integer", "description": "Maximum lines to return."},
        },
        "required": ["path"],
    },
)


def load_dotenv_key() -> str | None:
    """Read `OPENROUTER_API_KEY` from a `.env` in the repo root, if present."""

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip('"').strip("'")
    return None


async def stream_once(
    prompt: str,
    *,
    model: str,
    with_tools: bool,
    reasoning: str | None,
    api_key: str | None,
) -> int:
    """Stream one completion to stdout. Returns a process exit code."""

    provider = OpenRouterProvider(
        api_key=api_key,
        referer="https://github.com/azalabs/azalabscode",
        title="azalabscode",
    )
    request = ModelRequest(
        model=model,
        messages=[
            SystemMessage(content="You are concise. Answer in at most three sentences."),
            UserMessage.of(prompt),
        ],
        tools=[READ_FILE_SCHEMA] if with_tools else [],
        max_tokens=400,
        reasoning=ReasoningConfig(effort=reasoning) if reasoning else None,  # type: ignore[arg-type]
    )

    try:
        return await _run(provider, request, model)
    finally:
        # Everything that needs the transport, including the /models lookup, has to
        # happen inside _run: after this the client is closed.
        await provider.aclose()


async def _run(provider: OpenRouterProvider, request: ModelRequest, model: str) -> int:
    """Print the stream as it arrives, then the usage and metadata summary."""

    accumulator = StreamAccumulator(model=model)
    in_reasoning = False

    async for event in provider.stream(request):
        accumulator.feed(event)
        match event:
            case ReasoningDelta():
                if not in_reasoning:
                    print("\n[reasoning] ", end="", flush=True)
                    in_reasoning = True
                print(event.text, end="", flush=True)
            case TextDelta():
                if in_reasoning:
                    print("\n\n", end="", flush=True)
                    in_reasoning = False
                print(event.text, end="", flush=True)
            case ToolCallStart():
                print(f"\n[tool_call {event.index}] {event.name}(", end="", flush=True)
            case ToolCallDelta():
                print(event.arguments_delta, end="", flush=True)
            case ToolCallEnd():
                print(")", end="", flush=True)
            case Finish():
                print(f"\n\n[finish] {event.reason}")
            case UsageReport():
                pass
            case StreamError():
                print(f"\n\n[error] {event.error.kind}: {event.error.message}")

    if accumulator.error is not None:
        return 1

    usage = accumulator.usage
    if usage is not None:
        cost = f"${usage.cost_usd:.6f}" if usage.cost_usd is not None else "unknown"
        print(
            f"[usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"cached={usage.cached_tokens} reasoning={usage.reasoning_tokens} cost={cost}"
        )
    else:
        print("[usage] not reported")

    info = await provider.model_info(model)
    if info is not None:
        print(f"[model] context={info.context_length} tools={info.supports_tools}")

    for call in [p for p in accumulator.parts() if p.type == "tool_call"]:
        status = "parsed" if call.ok else f"parse_error: {call.parse_error}"
        print(f"[tool_call] {call.name} {call.arguments} ({status})")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Explain a ULID in one sentence.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tools", action="store_true", help="offer a read_file tool")
    parser.add_argument("--reasoning", choices=["minimal", "low", "medium", "high"])
    args = parser.parse_args(argv)

    api_key = load_dotenv_key()
    return run_sync(
        stream_once(
            args.prompt,
            model=args.model,
            with_tools=args.tools,
            reasoning=args.reasoning,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
