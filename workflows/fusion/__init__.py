"""Model-fusion reference workflow (spec 9.2, R-A-2).

The graph lands at M5 -- it is spec 6.3's own example and the thing M5's exit test
runs headless. The TUI (`SplitPanes` of `StreamPane`, `StagePipeline`) is M6.
"""

from workflows.fusion.workflow import (
    ANALYST_PROMPT,
    BRANCH_PROMPT,
    CONFIG_TYPE,
    IMPORT_PATH,
    SYNTHESIS_PROMPT,
    FusionConfig,
    build,
    slugify,
)

__all__ = [
    "ANALYST_PROMPT",
    "BRANCH_PROMPT",
    "CONFIG_TYPE",
    "IMPORT_PATH",
    "SYNTHESIS_PROMPT",
    "FusionConfig",
    "build",
    "slugify",
]
