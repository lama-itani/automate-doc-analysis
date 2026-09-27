# Status table — deferred hardening items (M-1)

These were identified during the M-1 build and **deliberately deferred**. The current
code is pilot-adequate for sequential / single-writer use. The items below matter for the
target architecture (many concurrent per-document CML Job invocations + an app orchestrator
all writing to one SQLite file) and should be closed before robust concurrent pilot use or
production. Ranked by importance.

| # | Item | Severity | Where | Fix sketch |
|---|---|---|---|---|
| 1 | `journal_mode` is `delete` (rollback journal), not **WAL**. Under concurrent multi-process writers, WAL greatly reduces reader/writer blocking. | High | `db.connect()` | `PRAGMA journal_mode=WAL;` + set `PRAGMA busy_timeout` explicitly (currently 5000 ms only via Python's implicit `timeout=5.0`). |
| 2 | Read-modify-write **race** in `transition`: current stage is read (`get_or_raise`) *outside* the write transaction, then `UPDATE`. Two concurrent writers to the *same* (case, doc) row → lost update / decision on stale state. Low real risk under the one-owner-per-document invariant. | Medium (latent) | `store.StatusStore.transition` | Do read+update in one `BEGIN IMMEDIATE` txn, or conditional `UPDATE ... WHERE stage = <expected>` and check `rowcount`. |
| 3 | `error_detail` is **optional** even when transitioning to `FAILED`. Conflicts with the "no silent failures" mandate. | Low–Medium | `store.StatusStore.transition` | Require non-empty `error_detail` when `target is FAILED`; raise otherwise. |
| 4a | `check_same_thread=True` (sqlite3 default). Fine for per-process connections, but a threaded orchestrator (M-3) sharing one connection will error. | Low | `db.connect()` | Document the per-process-connection assumption now; revisit at M-3. |
| 4b | `last_updated` at 1-second precision → possible ordering ties in the audit trail. | Low | `store._utcnow_iso` | Use millisecond precision if finer audit ordering is needed. |
| 4c | No logging yet (errors raise, which covers the critical "not silent" requirement). | Low | — | Add structured logging when the app/job layers land. |

_Status: all deferred by decision on 2026-09-27. Re-evaluate before M-3 (orchestrator) and
before ENS-High audit readiness (which is also the SQLite → hardened-store migration trigger)._
