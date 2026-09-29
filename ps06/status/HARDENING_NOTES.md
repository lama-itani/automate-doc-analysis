# Status table — deferred hardening items (M-1)

These were identified during the M-1 build and **deliberately deferred**. The current
code is pilot-adequate for sequential / single-writer use. The items below matter for the
target architecture (many concurrent per-document CML Job invocations + an app orchestrator
all writing to one SQLite file) and should be closed before robust concurrent pilot use or
production. Ranked by importance.

| # | Item | Severity | Where | Fix sketch |
|---|---|---|---|---|
| 1 | ~~`journal_mode` is `delete` (rollback journal), not **WAL**~~. **CLOSED (M-3, session 6):** `db.connect()` now sets `PRAGMA journal_mode=WAL;` + an explicit `PRAGMA busy_timeout=5000;` (`BUSY_TIMEOUT_MS`). Tested: `test_wal_journal_mode_file_backed`, `test_busy_timeout_set`. | ~~High~~ Done | `db.connect()` | — |
| 2 | ~~Read-modify-write **race** in `transition`~~. **CLOSED (M-3, session 6):** `transition` now applies a compare-and-swap `UPDATE ... WHERE ... AND stage = <validated stage>` and raises `TransitionConflict` on a 0-row result, so a concurrent stage change is surfaced, never a silent lost update. Tested: `test_transition_conflict_when_stage_changed_concurrently`. | ~~Medium (latent)~~ Done | `store.StatusStore.transition` | — |
| 3 | `error_detail` is **optional** even when transitioning to `FAILED`. Conflicts with the "no silent failures" mandate. | Low–Medium | `store.StatusStore.transition` | Require non-empty `error_detail` when `target is FAILED`; raise otherwise. |
| 4a | `check_same_thread=True` (sqlite3 default). Fine for per-process connections, but a threaded orchestrator (M-3) sharing one connection will error. | Low | `db.connect()` | Document the per-process-connection assumption now; revisit at M-3. |
| 4b | `last_updated` at 1-second precision → possible ordering ties in the audit trail. | Low | `store._utcnow_iso` | Use millisecond precision if finer audit ordering is needed. |
| 4c | No logging yet (errors raise, which covers the critical "not silent" requirement). | Low | — | Add structured logging when the app/job layers land. |

_Status: deferred by decision on 2026-09-27. Items #1 and #2 CLOSED during M-3 (session 6,
2026-09-29) as prerequisites for the orchestrator's concurrent per-document writers.
Remaining items (#3, #4a–c) still deferred; re-evaluate before ENS-High audit readiness
(which is also the SQLite → hardened-store migration trigger)._
