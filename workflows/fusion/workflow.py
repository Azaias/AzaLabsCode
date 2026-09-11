"""Model fusion: ask N models the same question, analyse the answers, synthesize one.

Spec 9.2, and the graph is spec 6.3's own example:

    models  ──fan out──▶  join  ──▶  analyze  ──▶  synthesize

`build(config)` is the whole contract (R-W-1). The session stores
`("workflows.fusion.workflow:build", config)` and nothing else -- no provider, no
client, no closure -- so another process can rebuild this graph from a JSON document
and resume it. That is also why **the providers are constructed in here** (spec
delta 21): a provider passed in from outside would be a piece of the run that the
file cannot describe.

**Each branch gets its own provider in fake mode.** `FakeProvider` hands out unkeyed
turns in order across the whole run, not per agent, so four branches sharing one
script interleave by whichever coroutine gets scheduled first and the fan-out looks
non-deterministic. One provider per branch removes the shared state entirely, which
is what makes the headless run reproducible enough to assert on.

Node ids are `models`, `models/<slug>`, `join`, `analyze`, `synthesize`. The slug
comes from the model name rather than its index because an index shifts when a model
is added to the middle of the list, and every session saved before that stops lining
up with the graph.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from azalabscode import (
    FakeProvider,
    ModelCall,
    OpenRouterProvider,
    Provider,
    ScriptedTurn,
    Workflow,
)

ANALYST_PROMPT = """You are comparing several independent answers to one question.

State where the answers agree, where they disagree and what each one has that the
others do not. Be specific about the disagreements: name the claim, name which
answers make it, and say which is better supported. Do not write a new answer to the
question and do not rank the models."""

SYNTHESIS_PROMPT = """You are writing the final answer to a question, given several
independent attempts at it and an analysis of where they differ.

Write the answer, not a review of the attempts. Take the best-supported claim
wherever they disagree, keep anything one attempt had that the others missed, and
leave out anything the analysis found unsupported. Do not mention the attempts, the
analysis or the process."""

BRANCH_PROMPT = "Answer the question directly and completely. Be concrete."


class FusionConfig(BaseModel):
    """Everything `build` needs, and everything the session stores (R-W-1)."""

    model_config = {"extra": "forbid"}

    question: str = ""
    """The workflow's input. Part of the config so a resumed run asks the same thing."""
    models: list[str] = Field(default_factory=list)
    analyst_model: str = ""
    synth_model: str = ""
    branch_prompt: str = BRANCH_PROMPT
    analyst_prompt: str = ANALYST_PROMPT
    synth_prompt: str = SYNTHESIS_PROMPT
    max_tokens: int | None = None
    temperature: float | None = None

    fake: dict[str, str] | None = None
    """Model id -> canned answer. When set, no network: every `ModelCall` gets its
    own `FakeProvider`. This is what `--headless` under test uses, and what makes the
    pause/save/load/resume assertions checkable at all (spec C-1)."""
    fake_chunk_delay_s: float = 0.0
    """Seconds between chunks in fake mode, so a test can pause mid-fan-out."""
    branch_concurrency: int = 0
    """How many branches may run at once; 0 is unbounded. Turned down when a caller
    is rate-limited, or when a test needs the pause to land somewhere it can name."""

    def branches(self) -> list[tuple[str, str]]:
        """`(slug, model)` per branch, with collisions disambiguated by index."""

        seen: dict[str, int] = {}
        out: list[tuple[str, str]] = []
        for model in self.models:
            slug = slugify(model)
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            out.append((slug if count == 0 else f"{slug}-{count}", model))
        return out


def slugify(model: str) -> str:
    """A model id as one path segment: `anthropic/claude-haiku-4.5` -> `claude-haiku-4-5`.

    The vendor prefix is dropped because it is the same for every model from one
    vendor and the node id is already scoped by `models/`.
    """

    tail = model.rsplit("/", 1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")
    return slug or "model"


def build(config: FusionConfig | dict[str, Any]) -> Workflow:
    """The importable factory (R-W-1). Returns spec 6.3's graph."""

    cfg = config if isinstance(config, FusionConfig) else FusionConfig.model_validate(config)
    shared = None if cfg.fake is not None else OpenRouterProvider()

    def provider_for(model: str) -> Provider | None:
        if cfg.fake is None:
            return shared
        return FakeProvider(
            [
                ScriptedTurn(
                    text=cfg.fake.get(model, f"(no scripted answer for {model})"),
                    chunk_delay_s=cfg.fake_chunk_delay_s,
                )
            ]
        )

    wf = Workflow("fusion", input=cfg.question)

    wf.fan_out(
        "models",
        {
            slug: ModelCall(
                model,
                provider=provider_for(model),
                system_prompt=cfg.branch_prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            for slug, model in cfg.branches()
        },
        max_concurrency=cfg.branch_concurrency,
    )
    joined = wf.gather("join", wf.ref("models"))
    analysis = wf.node(
        "analyze",
        ModelCall(
            cfg.analyst_model,
            provider=provider_for(cfg.analyst_model),
            system_prompt=cfg.analyst_prompt,
            render=render_answers,
        ),
        input=joined,
    )
    final = wf.node(
        "synthesize",
        ModelCall(
            cfg.synth_model,
            provider=provider_for(cfg.synth_model),
            system_prompt=cfg.synth_prompt,
            render=render_synthesis,
        ),
        input=(joined, analysis),
    )
    wf.output(final)
    return wf


def render_answers(value: Any) -> str:
    """The analyst's prompt: the branch answers, numbered."""

    answers = value if isinstance(value, list) else [value]
    body = "\n\n".join(f"--- answer {index + 1} ---\n{text}" for index, text in enumerate(answers))
    return f"Here are {len(answers)} independent answers to the same question.\n\n{body}"


def render_synthesis(value: Any) -> str:
    """The synthesizer's prompt: the answers, then the analysis."""

    answers, analysis = value if isinstance(value, tuple) else (value, "")
    return f"{render_answers(answers)}\n\n--- analysis ---\n{analysis}"


IMPORT_PATH = "workflows.fusion.workflow:build"
CONFIG_TYPE = "workflows.fusion.workflow:FusionConfig"

__all__ = [
    "ANALYST_PROMPT",
    "BRANCH_PROMPT",
    "CONFIG_TYPE",
    "IMPORT_PATH",
    "SYNTHESIS_PROMPT",
    "FusionConfig",
    "build",
    "render_answers",
    "render_synthesis",
    "slugify",
]
