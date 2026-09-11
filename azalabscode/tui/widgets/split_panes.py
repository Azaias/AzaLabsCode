"""`SplitPanes`: N panes side by side, or in a grid when N gets large (R-U-3).

Spec 8.2 gives this widget an empty "consumes" column, and that is the whole design:
it is a layout, not a consumer. The panes inside it are the consumers, and they are
registered with the router individually because `EventRouter` walks to the outermost
`EventWidget` in each branch of the tree -- a container that was itself an
`EventWidget` would swallow the panes' registrations.

The only judgement in here is when to stop putting panes in one row. Four 20-column
panes on an 80-column terminal is four unreadable columns, so past `max_columns` the
layout wraps into a grid. Fusion at four models is the case this exists for, and M4
measured why it matters that the panes are `tail=True`: a fixed-size pane never asks
Textual to re-arrange the screen, and re-arranging is what costs.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from textual.containers import Container
from textual.widget import Widget

MAX_COLUMNS = 3
"""Panes per row before the layout wraps. Below about 26 columns a streaming pane
is a column of single words."""


class SplitPanes(Container):
    """A grid of panes, sized to fit them (R-U-3).

    Not an `EventWidget`: the panes it holds are the consumers, and the router finds
    them by walking the tree.
    """

    DEFAULT_CSS = """
    SplitPanes {
        layout: grid;
        height: 1fr;
        width: 1fr;
    }
    SplitPanes > Widget {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(
        self,
        *panes: Widget,
        columns: int = 0,
        max_columns: int = MAX_COLUMNS,
        **kwargs: object,
    ) -> None:
        super().__init__(*panes, **kwargs)  # type: ignore[arg-type]
        self.max_columns = max_columns
        self._requested_columns = columns
        self._panes: list[Widget] = list(panes)

    @property
    def panes(self) -> list[Widget]:
        """The panes, in layout order."""

        return list(self._panes)

    def columns_for(self, count: int) -> int:
        """How many columns `count` panes get."""

        if self._requested_columns:
            return max(1, self._requested_columns)
        return max(1, min(count, self.max_columns))

    def on_mount(self) -> None:
        """Size the grid to the panes it was given."""

        self.resize_grid()

    def resize_grid(self) -> None:
        """Recompute rows and columns. Called after any change to the pane list."""

        count = len(self._panes) or 1
        columns = self.columns_for(count)
        self.styles.grid_size_columns = columns
        self.styles.grid_size_rows = max(1, math.ceil(count / columns))

    async def add_pane(self, pane: Widget) -> None:
        """Mount one more pane and re-size the grid."""

        self._panes.append(pane)
        await self.mount(pane)
        self.resize_grid()

    async def set_panes(self, panes: Iterable[Widget]) -> None:
        """Replace every pane. Used when a layout is rebuilt for a loaded run."""

        await self.remove_children()
        self._panes = list(panes)
        await self.mount_all(self._panes)
        self.resize_grid()


__all__ = ["MAX_COLUMNS", "SplitPanes"]
