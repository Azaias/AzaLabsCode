"""`CodingSession`: the coding agent as a thing you can talk to.

The TUI and the headless CLI both drive this, so "send a message" means the same in
both and there is one place where the two verbs of spec 8.2 are implemented:

* **`send(text)`** injects and wakes. It does not cancel: the agent finishes the step
  it is on and reads the message at its next turn boundary (R-C-4). If the agent is
  idle between prompts, the wake-up is what starts it again.
* **`interrupt(text)`** cancels the current step first and then injects, which is what
  `escape` does in the TUI.

Both go through the `Controller`, so both emit `MessageInjected` and both survive a
save: the message is on `AgentState.pending_injections` before anything else happens
to it.

`close()` ends the session after the current prompt. Without it an interactive run
never returns, which is correct for a session and wrong for a process that is trying
to exit -- so the CLI calls it, and so does the TUI on unmount.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from azalabscode import (
    Controller,
    EventBus,
    PermissionMode,
)
from workflows.coding_agent.workflow import (
    AGENT_ID,
    CONFIG_TYPE,
    IMPORT_PATH,
    CodingAgentConfig,
    agent_node,
    build,
)
from workflows.tooling import ToolingAgentNode


class CodingSession:
    """A bound controller, its agent node, and the two ways to talk to it."""

    def __init__(
        self,
        controller: Controller,
        node: ToolingAgentNode,
        config: CodingAgentConfig,
    ) -> None:
        self.controller = controller
        self.node = node
        self.config = config

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        config: CodingAgentConfig | dict[str, Any],
        *,
        session_dir: str | Path | None = None,
        bus: EventBus | None = None,
        mode: PermissionMode = PermissionMode.MANUAL,
        approval_handler: Any = None,
        autosave: bool = True,
        run_id: str | None = None,
    ) -> CodingSession:
        """A fresh session with the graph bound and its rebuild recipe recorded.

        `manual` is the default mode, which is `permissions.DEFAULT_MODE` and the safe
        direction: a mistaken prompt costs a keystroke, a mistaken `shell` costs a
        repository. `--auto` is how a caller says otherwise.
        """

        cfg = (
            config
            if isinstance(config, CodingAgentConfig)
            else CodingAgentConfig.model_validate(config)
        )
        controller = Controller(
            run_id=run_id,
            bus=bus,
            permission_mode=mode,
            approval_handler=approval_handler,
            session_dir=session_dir,
            autosave=autosave,
            main_agent=AGENT_ID,  # type: ignore[arg-type]
        )
        workflow = build(cfg)
        controller.bind_workflow(
            workflow,
            import_path=IMPORT_PATH,
            config=cfg.model_dump(mode="json"),
            config_type=CONFIG_TYPE,
        )
        return cls(controller, agent_node(controller), cfg)

    @classmethod
    async def load(
        cls,
        path: str | Path,
        *,
        bus: EventBus | None = None,
        approval_handler: Any = None,
    ) -> CodingSession:
        """Reopen a saved session. The graph is rebuilt from `(import_path, config)`."""

        controller = await Controller.load(
            path,
            bus=bus if bus is not None else EventBus(),
            approval_handler=approval_handler,
        )
        cfg = CodingAgentConfig.model_validate(controller.workflow.config)
        return cls(controller, agent_node(controller), cfg)

    # -- talking to it ------------------------------------------------------

    async def send(self, text: str) -> str | None:
        """Queue a message and wake the session. Returns the message id, or `None`.

        Non-cancelling by design (spec 8.2's normal mode). The wake-up is sent whether
        or not the injection landed: if the agent is between prompts there is nothing
        to inject into yet, and the node re-checks after every wake.
        """

        message_id = await self.controller.send(text, target=AGENT_ID)
        self.node.wake()
        return message_id

    async def interrupt(self, text: str | None = None) -> Any:
        """Cancel the agent's current step, then optionally inject (`escape`)."""

        result = await self.controller.interrupt(text, target=AGENT_ID)
        self.node.wake()
        return result

    def close(self) -> None:
        """End the session once the current prompt is answered."""

        self.node.close()

    # -- running ------------------------------------------------------------

    async def start(self) -> None:
        """Begin the run."""

        await self.controller.start()

    async def wait(self, *, timeout: float | None = None) -> str:  # noqa: ASYNC109 - the bound is the caller's; Controller.wait takes it
        """Wait for the run to finish and return the agent's last answer."""

        return str(await self.controller.wait(timeout=timeout) or "")

    async def run_once(self, *, timeout: float | None = None) -> str:  # noqa: ASYNC109 - the bound is the caller's; Controller.run takes it
        """Start, answer one prompt, and return -- the `--headless` path (R-U-7)."""

        if self.config.interactive:
            self.close()
        return str(await self.controller.run(timeout=timeout) or "")


__all__ = ["CodingSession"]
