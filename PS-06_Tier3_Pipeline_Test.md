# Tier-3 Live Test: Full Pipeline (Ingestion → OCR/Classification → Rules) on One Folder

## Context
Ingestion (`--folder`, shipped this session), classification (M-2.5), and the
rules/aggregation engine (M-4, `--evaluate-rules`) have each been validated
individually — classification was live-tested on a real 4-PDF folder in Tier-2
(`PS-06_Tier2_Classification_Folder_Test.md` / `_Results.md`, 4/4 correct), and
the rules engine is unit-tested (277+ tests) but **has never run against a real
Cloudera endpoint** — only against pre-seeded/fixture extractions
(`test_rules_engine.py`, `test_orchestrator_rules_e2e.py`). The user now has a
Workbench session and a folder of 4 PDFs, and wants to confirm the full chain —
folder scan → OCR/VLM → classification → rules/triage (VERDE/AMARILLO/ROJO) —
works end-to-end in one call.

This is a **runbook, not a code change** — same spirit as Tier-1/Tier-2. It
exercises `ps06.orchestrator.cli run-case --folder ... --evaluate-rules`, which
composes entirely of already-shipped, already-tested code
(`ps06/orchestrator/intake.py`, `ps06/orchestrator/orchestrator.py::process_case`,
`ps06/rules/engine.py::evaluate_case`).

## Constraint (unchanged from Tier-1/Tier-2, reconfirmed by code inspection)
`CDSWAuthProvider.mint_token()` (`ps06/ocr/auth.py:37-48`) and
`WorkbenchJobLauncher.launch()` (`ps06/orchestrator/launcher.py:214-226`) are
both **honest `NotImplementedError` stubs** — real Knox JWT minting and a real
per-document CML Job launch are not implemented (Build Handoff open item #4).
`--auth cdsw` fails predictably (every doc ends `FAILED` with a
`NotImplementedError` `error_detail` — not a crash). So this test, like Tier-1/2,
uses `--auth fake` with a **real, hand-obtained JWT** from the Workbench
session — everything past auth (HTTP client, extraction, VLM, classification,
rules engine, status writes) is real production code; only token minting is
faked. `LocalThreadJobLauncher` (the only working launcher) is what
`orchestrator/cli.py` always uses, so no launcher flag is needed.

## Runbook

### Step 1 — Confirm the Workbench checkout is current
Tier-2's results log recorded a real incident: an out-of-date checkout raised
`ModuleNotFoundError: ps06.orchestrator`. Before anything else:
```bash
git log --oneline -1
git status
```
Pull if the `--folder`/`--evaluate-rules` work isn't present (`ps06/orchestrator/intake.py`
should exist). Confirm which venv has the right dependencies (Tier-2 used
`venv310`; check both `venv310`/`venv311` exist and pick the one importing cleanly):
```bash
./venv311/bin/python -c "import ps06.orchestrator.cli, ps06.orchestrator.intake, ps06.rules.engine" || \
./venv310/bin/python -c "import ps06.orchestrator.cli, ps06.orchestrator.intake, ps06.rules.engine"
```
Use whichever venv succeeds for every command below (substitute as `$PY`).

### Step 2 — Get a real JWT + endpoint details (same as Tier-1/Tier-2)
```bash
export JWT=$(python -c 'import json;print(json.load(open("/tmp/jwt"))["access_token"])')
```
Derive `--endpoint-url` by stripping the model endpoint's last two path segments
down to the `.../v1` root; note `--model-name` exactly as served (Tier-2 used
`Qwen/Qwen2.5-VL-7B-Instruct`).

### Step 3 — Init a fresh, scoped status DB
```bash
$PY -m ps06.status.cli --db pipeline_test.db init-db
```
No manual `add-doc` needed — `--auto-register` (Step 4) registers all
folder-discovered documents at `RECEIVED`.

### Step 4 — Run the whole folder as one case, OCR → classification → rules, in one call
This is the point of this test: add `--evaluate-rules` (and `--folder` instead
of four `--doc` pairs) to what Tier-2 ran manually per-doc.
```bash
$PY -m ps06.orchestrator.cli --db pipeline_test.db run-case \
  --case folder_test1 \
  --folder /path/to/your/4pdf/folder \
  --endpoint-url "$ENDPOINT_URL" --model-name "$MODEL_NAME" \
  --auth fake --fake-token "$JWT" \
  --max-concurrency 2 \
  --auto-register \
  --evaluate-rules \
  --json
```
- Orientation correction stays off (default) — known live finding (Tier-1) that
  it can silently corrupt scanned images.
- `--folder` derives each document's id from its filename stem (e.g.
  `Pasaporte.pdf` → id `Pasaporte`) — rename files beforehand if clearer ids are
  wanted for reading Step 5's output, or accept the stems as-is.
- If the case folder is one applicant (matches M-4's expected shape — one
  Solicitud, one ID, up to two/three birth certs) rules evaluation is meaningful;
  if it's 4 unrelated documents, expect `AMARILLO` (missing expected docs / no
  corroboration) or an entity-resolution flag — not a bug, see Step 6.
- Drop `--json` if Workbench session logs persist and the docs are sensitive
  (prints full extracted text to output).

Expected terminal states: `run-case` prints a `CaseSummary` — confirm
`[4/4 OCR_DONE]` and no `FAILED`. If `--evaluate-rules` fired, the case's
derived status becomes `RULES_EVALUATED`; if any doc `FAILED`, the rules hook is
skipped entirely (never runs on a partial case) — check `error_detail` per
failed doc instead of expecting a snapshot.

### Step 5 — Inspect each stage's output
```bash
# Orchestrator/OCR view: per-doc stage + case-level derived status
$PY -m ps06.orchestrator.cli --db pipeline_test.db status --case folder_test1

# Per-doc OCR + classification detail (repeat --doc per discovered id)
$PY -m ps06.ocr.cli --db pipeline_test.db show --case folder_test1 --doc <id> --json

# Rules/triage output: VERDE/AMARILLO/ROJO + justification + S1-S3 rows
$PY -m ps06.rules.cli --db pipeline_test.db show-case --case folder_test1 --json
```
`ps06.ocr.cli show --json`'s `classification` field carries `document_type`
(and `generation` for birth certs). `ps06.rules.cli show-case` prints `estado`,
`justificacion`, `problemas`, plus s1 (doc inventory), s2 (per-doc eval —
`campos_faltantes`, `consistencia`), s3 (lineage) rows.

### Step 6 — Judge correctness against ground truth
- **Classification** (per Tier-2's pass criteria): application form →
  `APPLICATION`, passport/ID → `ID_DOCUMENT`, birth certs → `BIRTH_CERT`. Flag
  any `OTHER`/`EMPTY` and check `raw_response`.
- **Rules/triage**: cross-check the printed `estado` against the actual
  document set using the M-4 Source Logic Reference (Build Handoff doc) —
  5 expected documents for `VERDE`, `DOCUMENTO_VENCIDO`/`NOMBRE_INCONSISTENTE`/
  etc. for `ROJO`, missing-doc/`DOCUMENT_INCOMPLETE`/insufficient corroboration
  for `AMARILLO`. With only 4 docs (5 expected for VERDE) and no G2/G3
  documents populated by the adapter yet, landing `AMARILLO` is the expected
  outcome per Binding Decision #6, not a defect — a `ROJO` verdict is the
  interesting signal to scrutinize (check `problemas` for which flag fired and
  whether it's a real defect vs. a genuine document issue).
- Record any mismatch with the relevant `raw_response` / `justificacion` text —
  this is the same "real-world signal vs. code defect" judgment Tier-2 used.

## Data handling (same as Tier-1/Tier-2 — mandatory)
`pipeline_test.db` holds full extracted applicant text in plaintext
(`document_extraction.payload`) plus the rules snapshot. Keep it in a scoped,
uncommitted location; delete (`pipeline_test.db`, `-wal`, `-shm`) after the test.
Drop `--json` if session logs persist. `$JWT` on the command line is visible via
`ps` — acceptable on a single-tenant session only.

## What this validates / does not
**Validates:** the full `--folder` → OCR/VLM → classification → rules pipeline
in one orchestrator call, against a real endpoint, for the first time (rules
engine has never run against live extractions before). **Does not validate:**
real Knox token minting or a real Workbench-API-v2 job launch (both open item
#4 stubs, confirmed unconditional `NotImplementedError` in
`ps06/ocr/auth.py`/`ps06/orchestrator/launcher.py`); reliability across
multiple folders (this is one folder, 4 docs); lineage/generation resolution
correctness beyond what one folder's document set can exercise.

## No code changes
Pure runbook. If classification or rules output looks wrong on real documents,
that's a follow-up finding (prompt tuning / rules scoping), not something to
fix inline during the test.

## Re-run (Fix-plan item 7)
Same procedure, same folder/endpoint. Deltas only:

- **Step 1:** confirm the commit with fix-plan items 2-5 (Anexo field mapping,
  firma detection, structured field extraction, generation resolution) is
  present (`git log --oneline -1`), not just `--folder`/`--evaluate-rules`.
- **Step 3:** fresh `pipeline_test2.db` — don't reuse the first run's DB.
- **Step 4:** case id `folder_test2` (distinguish from the first run).
- **Step 6, check against the first run's findings** (`PS-06_Tier3_Results.md`):
  - s1: G1/G2 cert rows now **Presente** (were Faltante) — finding #1 fixed.
  - s3: G1/G2 lineage rows resolved (were "No determinado") — same.
  - Anexo 4 fields populated in s1/s2 (was all-missing) — finding #2 fixed.
  - `campos_faltantes` on ID/certs reflects real extracted values, not
    blanket-missing — finding #3 fixed.
  - `firma` reflects the page-2 "SIGN" token.
  - `estado` may differ from AMARILLO; record whatever it is + `justificacion`.
    No G3 cert in this folder, so AMARILLO from the genuinely missing G3 is
    still a correct outcome, not a defect (finding #4).
- **Output:** write `PS-06_Tier3_Results_2.md` in the same format as the
  original — only list genuinely new findings; confirm findings 1-3 resolved
  rather than re-describing them.
- **Cleanup:** delete `pipeline_test2.db`/`-wal`/`-shm` after review.