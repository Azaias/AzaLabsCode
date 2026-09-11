"""The default key bindings (spec 8.1), in one table a subclass can edit.

Kept out of `app.py` so a workflow-specific app can start from this list and drop
or remap entries without restating the ones it keeps. `HarnessApp.BINDINGS` is this
list; every action it names is defined on `HarnessApp`.

`ctrl+c` is bound with `priority=True` on purpose. Textual binds it to its own
`help_quit` as a system binding, and a run that quits the UI without cancelling the
controller leaves an agent spending money with nobody watching. Two presses inside
`DOUBLE_PRESS_WINDOW_S` cancel the run (spec 8.1); one press only warns.
"""

from __future__ import annotations

from textual.binding import Binding, BindingType

DOUBLE_PRESS_WINDOW_S = 2.0
"""How long the first `ctrl+c` stays armed. Long enough to be deliberate, short
enough that a press two minutes later is not read as the second half of a pair."""

HARNESS_BINDINGS: list[BindingType] = [
    Binding("ctrl+p", "toggle_pause", "Pause", show=True),
    Binding("escape", "interrupt", "Interrupt", show=True),
    Binding("ctrl+t", "toggle_permission_mode", "Mode", show=True),
    Binding("ctrl+s", "save_session", "Save", show=True),
    Binding("ctrl+o", "open_session", "Open", show=True),
    Binding("ctrl+l", "toggle_event_log", "Log", show=True),
    Binding("ctrl+c", "request_cancel", "Cancel run", show=True, priority=True),
]
"""Spec 8.1's table. `y`/`n` are bound by `ApprovalModal`, not here: they are
ordinary text everywhere else and a global binding would eat them."""


def forwarded_to_app(*, drop: frozenset[str] = frozenset()) -> list[BindingType]:
    """`HARNESS_BINDINGS` re-pointed at the app's namespace, for a modal screen.

    Textual stops binding lookup at a `ModalScreen`: a key pressed while a modal is
    up never reaches `App.BINDINGS`. That is the right default -- a modal is modal
    -- and it is wrong for every binding in spec 8.1's table, because the state a
    user most wants to switch permission mode or save from is exactly the one where
    a modal is up. M3 found the same thing about `save()`: `WAITING_APPROVAL` is the
    state a user is most likely to save from, and it was the one state that did not
    work.

    Re-pointing the action at the `app.` namespace makes Textual resolve it on the
    `App` rather than on the screen, so the modal borrows the app's actions without
    redeclaring any of them.

    `drop` is for keys the modal needs for itself -- `escape` denies there.
    """

    return [
        Binding(
            binding.key,
            f"app.{binding.action}",
            binding.description,
            show=binding.show,
            priority=binding.priority,
        )
        for binding in HARNESS_BINDINGS
        if isinstance(binding, Binding) and binding.key not in drop
    ]


__all__ = ["DOUBLE_PRESS_WINDOW_S", "HARNESS_BINDINGS", "forwarded_to_app"]
