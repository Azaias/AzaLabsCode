"""The run lifecycle: controller, state machine, quiescence, gate, approvals.

M2 landed the lifecycle in memory; M3 added the durable half. `session.py` is the
document, `checkpoint.py` decides where it goes, `atomic.py` guarantees a reader
never sees a half-written one, and `resume.py` says what a loaded session means.

This layer implements two of the four inversion protocols in
`azalabscode.contracts`: `RunControl` (consumed by the agent loop) and
`PermissionGate` (consumed by `tools.dispatcher`). Neither of those layers imports
this one.
"""

from azalabscode.control.approval_handlers import (
    CallbackApprovalHandler,
    DenyAllHandler,
    QueueApprovalHandler,
    StdinApprovalHandler,
)
from azalabscode.control.atomic import (
    atomic_write_bytes,
    atomic_write_bytes_async,
    atomic_write_text,
    sweep_temp_files,
)
from azalabscode.control.checkpoint import Checkpointer
from azalabscode.control.controller import (
    DEFAULT_SAVE_TIMEOUT,
    Controller,
    FoldRecord,
    InterruptResult,
    RunBody,
)
from azalabscode.control.gate import RuntimePermissionGate, gated_names
from azalabscode.control.quiescence import QuiescenceTracker
from azalabscode.control.resume import (
    ResumeReport,
    check_graph_drift,
    reconcile,
    resume_state_for,
)
from azalabscode.control.session import (
    InflightStep,
    NodeRecord,
    Session,
    ValueRef,
    WorkflowRef,
)
from azalabscode.control.state import AgentState, IllegalTransition, RunStateMachine

__all__ = [
    "DEFAULT_SAVE_TIMEOUT",
    "AgentState",
    "CallbackApprovalHandler",
    "Checkpointer",
    "Controller",
    "DenyAllHandler",
    "FoldRecord",
    "IllegalTransition",
    "InflightStep",
    "InterruptResult",
    "NodeRecord",
    "QueueApprovalHandler",
    "QuiescenceTracker",
    "ResumeReport",
    "RunBody",
    "RunStateMachine",
    "RuntimePermissionGate",
    "Session",
    "StdinApprovalHandler",
    "ValueRef",
    "WorkflowRef",
    "atomic_write_bytes",
    "atomic_write_bytes_async",
    "atomic_write_text",
    "check_graph_drift",
    "gated_names",
    "reconcile",
    "resume_state_for",
    "sweep_temp_files",
]
