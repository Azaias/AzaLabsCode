"""The nine built-in tools (R-T-8, spec 7).

Each module holds one tool, its `Params` model, and its `DESCRIPTION` -- a
module-level constant reviewed like code and covered by the schema snapshot test
(R-T-9), so an accidental description edit shows up in a diff.

Import tools from `azalabscode.tools`, not from here: this package is an
implementation detail of the layout, and `default_registry()` is the supported way
to build the set.
"""

__all__: list[str] = []
