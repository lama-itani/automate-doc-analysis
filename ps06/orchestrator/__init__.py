"""App-layer orchestrator for PS-06 (M-3).

Launches one per-document OCR job invocation per document (via a
:class:`~ps06.orchestrator.launcher.JobLauncher`), polls the status table, and
retries ``FAILED`` documents as fresh invocations with bounded concurrency —
the "APPLICATION SERVING LAYER -> Orchestrator" node of the No-TTL architecture.
Reuses M-1 (:mod:`ps06.status`) and M-2 (:mod:`ps06.ocr`) verbatim.
"""

from ps06.orchestrator.launcher import (
    JobHandle,
    JobLauncher,
    LocalThreadJobLauncher,
    WorkbenchJobLauncher,
)
from ps06.orchestrator.orchestrator import (
    CaseSummary,
    DocumentOutcome,
    OrchestratorConfig,
    process_case,
)

__all__ = [
    "CaseSummary",
    "DocumentOutcome",
    "JobHandle",
    "JobLauncher",
    "LocalThreadJobLauncher",
    "OrchestratorConfig",
    "WorkbenchJobLauncher",
    "process_case",
]
