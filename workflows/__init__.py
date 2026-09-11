"""Reference workflows, built entirely on the public API.

Outside the `azalabscode` package on purpose: these are consumers, and
import-linter contract 5 proves it by forbidding them from importing any private
module. If a reference workflow needs something it cannot reach, the fix is to
export it, not to reach past the boundary.

`coding_agent` (R-A-1), `fusion` (R-A-2) and `inspector` (R-A-3) land at M6.
"""

__all__: list[str] = []
