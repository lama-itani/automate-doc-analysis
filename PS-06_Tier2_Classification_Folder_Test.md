# Tier-2 Live Test Plan: Document Classification on a Real Folder (M-2.5)

## Context

M-2.5 (document classification, closing Open Item #5) is complete in code and green
in the unit suite (130 passed), but it has **never been exercised against a live
Cloudera AI Inference endpoint** — the standing blocker is Open Item #4
(`CDSWAuthProvider.mint_token()` is still an honest `NotImplementedError` stub;
confirmed live in `ps06/ocr/auth.py:39,44`, both unconditionally raise regardless of
whether `CDSW_APIV2_KEY` is set). Only `--auth fake` with a hand-obtained real bearer
token can drive a live call today — the same workaround Tier-1 used.

The user now has a Workbench session up and running and wants to validate
classification end-to-end on a **realistic folder**: one applicant's case containing
multiple documents of different types (passport/ID, birth certificate(s), application
form) — the real-world shape the folder-agnostic per-document classifier was designed
around (per the session-7 Progress Log scoping notes in `PS-06_Build_Handoff.md`).

This extends the prior Tier-1 live test (`PS-06_Tier1_Workbench_Test.md` /
`PS-06_Tier1_Results.md`, run 2026-09-28) — which validated OCR/VLM/auth-retry but
predates classification — to cover classification specifically, across a whole
folder rather than one or two docs. Goal: confirm each document in the folder gets a
**plausible, correct `document_type`** (and `generation` where applicable), using the
real endpoint, real extracted text, and the real classification prompt.

## Approach: reuse `ps06.orchestrator.cli run-case`, one case = one folder

Use the orchestrator CLI (`ps06/orchestrator/cli.py`, built in M-3) rather than
driving each document one-by-one through `ps06.ocr.cli run` (Tier-1's method, before
the orchestrator existed) — it already exists, is unit-tested, and is designed for
exactly this: one case, many documents, one command, bounded concurrency, status
tracking. This also incidentally exercises the M-3 orchestrator against a live
endpoint for the first time.

**Known limitation:** `run-case`'s `_build_ocr_config` does not expose
`--classification-*` flags — it always uses `ClassificationConfig` defaults
(confirmed by direct inspection of `ps06/orchestrator/cli.py`). That's fine here
since this test validates classification behavior, not tuning; if classification
params ever need tuning, drive documents individually via `ps06.ocr.cli run`
instead, which does expose `--classification-max-tokens` /
`--classification-temperature` / `--classification-max-text-chars` /
`--classification-min-chars`.

`--auth cdsw` still cannot succeed — this test uses the same `--auth fake` +
hand-obtained real token pattern as Tier-1. Everything downstream of auth (HTTP
client, extraction, classification, status writes) is real production code; only
the token-minting mechanism is faked.

### Step 1 — Pick the test folder and get a real token
- Select one applicant folder with 3-6 documents of mixed types (at least one
  scanned/image-only doc so the VLM path — and therefore the endpoint and
  classification — is exercised; a pure text-layer PDF alone still gets classified,
  since `classify()` always calls the LLM once extracted text clears the 10-char
  EMPTY threshold, but including a scanned doc also re-validates the OCR path).
- Inside the Workbench session, obtain the JWT the same way as Tier-1:
  ```bash
  export JWT=$(python -c 'import json;print(json.load(open("/tmp/jwt"))["access_token"])')
  ```
- Derive `--endpoint-url` (strip the model endpoint's last two path segments to the
  `.../v1` root) and note `--model-name` exactly as served.

### Step 2 — Initialize a fresh status DB
```bash
./venv311/bin/python -m ps06.status.cli --db classify_test.db init-db
```
No manual `add-doc` needed — `run-case --auto-register` (Step 3) registers each
supplied document at `RECEIVED` automatically.

### Step 3 — Run the whole folder as one case via the orchestrator
Build one `--doc ID=PATH` per file in the folder (repeatable flag), pick conservative
concurrency (small folder — `--max-concurrency 2` is enough, stays comfortably
inside the ~1hr JWT TTL per Open Item #1), and leave orientation correction off
(default) per the still-standing Tier-1 finding that it can silently corrupt data:

```bash
./venv311/bin/python -m ps06.orchestrator.cli --db classify_test.db run-case \
  --case folder_test1 \
  --doc app=/path/to/folder/Solicitud.pdf \
  --doc id1=/path/to/folder/Pasaporte.pdf \
  --doc bc1=/path/to/folder/CertificadoNacimiento_1.pdf \
  --doc bc2=/path/to/folder/CertificadoNacimiento_2.pdf \
  --endpoint-url "$ENDPOINT_URL" --model-name "$MODEL_NAME" \
  --auth fake --fake-token "$JWT" \
  --max-concurrency 2 \
  --auto-register \
  --json
```
- Use meaningful `--doc` ids (as above) rather than `doc1`/`doc2` — makes the
  per-document classification results in Step 4 easy to eyeball against expectations
  without cross-referencing file paths.
- If a file is a text-layer PDF and you also want to re-validate the VLM/orientation
  path, add `--no-text-layer-fast-path`; not required for classification itself,
  since classification runs regardless of which OCR path was taken.
- Drop `--json` if Workbench session logs are retained and the documents are
  sensitive (same data-handling caveat as Tier-1 — `--json` prints full extracted
  text to session output).

### Step 4 — Inspect classification results per document
```bash
./venv311/bin/python -m ps06.orchestrator.cli --db classify_test.db status --case folder_test1
./venv311/bin/python -m ps06.ocr.cli --db classify_test.db show --case folder_test1 --doc app --json
./venv311/bin/python -m ps06.ocr.cli --db classify_test.db show --case folder_test1 --doc id1 --json
# ...repeat `show` per doc id (bc1, bc2, ...)
```
Each `show --json` payload's `classification` field carries `document_type`,
`generation` (only meaningful for `BIRTH_CERT`), `confidence` (placeholder, always
1.0 today — Req #11 not implemented yet), and `raw_response` (the model's raw
answer — useful for eyeballing *why* a doc was classified a certain way, especially
any `OTHER`/misclassification).

### Step 5 — Judge correctness against ground truth
Manually compare each document's known real type against the `document_type` the
run produced:
- **Pass:** application form → `APPLICATION`, passport/DNI/cédula → `ID_DOCUMENT`,
  birth certificates → `BIRTH_CERT` (with `generation` set only when the source text
  itself states a generation — it's expected for `generation` to be `None` if the
  document doesn't self-state it; generation resolution across a family's documents
  is M-4's job, not this classifier's).
- **Watch for:** any doc landing in `OTHER` (check `raw_response` — unparseable
  model output vs. a genuinely ambiguous document) or `EMPTY` (check whether that
  document's extracted text really was near-blank vs. an extraction problem
  upstream).
- Record any misclassification with its `raw_response` — this is exactly the
  real-world signal (Spanish-language cues, prompt wording) the classifier's prompt
  was designed around but has never been checked against real documents.

## Data-handling reminder (same as Tier-1, still applies)
`classify_test.db` holds full extracted text of real applicant documents in
plaintext (`document_extraction.payload`). Keep it in a scoped, uncommitted
location; delete after the test. Avoid `--json` output if session logs persist. The
`$JWT` on the command line is visible via `ps` — acceptable on a single-tenant
Workbench session, not on a shared host.

## What this does and does not validate
**Validates:** classification against real Spanish-language documents and a real
endpoint, for a realistic multi-document folder, using the actual orchestrator path
a real case would take. **Does not validate:** real Knox token minting (Open Item
#4, `CDSWAuthProvider` still a stub — this test uses `--auth fake` throughout), or
generation/lineage resolution (M-4, not built yet — `generation` being `None` across
the board on a birth-cert-heavy folder is expected, not a bug).

## No code changes
This is a pure runbook, like Tier-1 — no edits to `ps06/`. If classification proves
wrong or unreliable on real documents, that's a finding for a separate follow-up
session (prompt tuning, taxonomy adjustment), not something to fix inline during
this test.