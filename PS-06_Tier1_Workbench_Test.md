# Corrected Tier-1 Workbench Test Plan (M-2 OCR job)

## Context
Goal: exercise the built M-2 OCR/extraction pipeline against a **real Cloudera AI
Inference endpoint** on a real (non-prod) Workbench, using a few **approved** real
immigration documents. This validates everything that exists today; it does not
validate real token minting, because `CDSWAuthProvider.mint_token()`
(`ps06/ocr/auth.py:37`) is still an unconditional `NotImplementedError` stub
(Build Handoff open item #4).

An earlier draft of this test had four defects, confirmed against source. This plan
corrects them:
1. **Wrong retry claim.** A mid-run 401 with a single `--fake-token` raises
   `RuntimeError` from `_build_client()` (`http_client.py:75` → `auth.py:68`), **not**
   `AuthRetryExhausted`. Recovery is only exercised by seeding two tokens: a bad one
   then a good one.
2. **Fast-path skips the endpoint.** A PDF with a text layer takes the text-layer
   fast path and never calls the VLM, so the endpoint/auth go untested.
3. **DB-name inconsistency.** Draft mixed `ps06.db` and `ps06_status.db`; must be one.
4. **Re-run collision.** Re-running a doc already at `OCR_DONE` is an invalid
   transition → the job records `FAILED` (per session-4 `test_already_ocr_done_...`).
   Each run needs a fresh `--doc` id.

Decisions from this session: data is **real + approved**; the run **does** include the
deliberate 401-recovery test (recommended — it is the mechanism the rebuild exists to fix).

## Constraint that shapes everything
`--auth cdsw` cannot succeed today (stub raises). Tier 1 uses `--auth fake` with a
**real** token obtained by hand inside the Workbench session — a legitimate use of the
test seam, since everything past auth (HTTP client, retry control flow, extraction, VLM
calls, status writes) is the real code path.

## Pre-run setup (inside the Workbench session)

1. **Get a real token.** Per Cloudera docs the session token is on disk:
   ```bash
   export JWT=$(python -c 'import json;print(json.load(open("/tmp/jwt"))["access_token"])')
   ```
   Confirm `/tmp/jwt` exists and this yields a non-empty string. (This is also the likely
   basis for a future real `mint_token()` — note its TTL and whether prod Workbench
   provides `/tmp/jwt` too; both are open questions.)
2. **Derive the base URL.** Take the model's endpoint URL and strip the last two path
   components so it ends at the API version (`.../v1`). This is `--endpoint-url`.
3. **Note the model name** exactly as served → `--model-name`.
4. **Env/deps.** Use the repo's `venv311` (openai 3.19.2, pymupdf 1.28.2, pillow, pydantic).
5. **Pick documents.** Choose at least one **scanned / image-only** doc (no text layer) so
   the VLM path — and therefore the endpoint and auth — is actually exercised. A text-layer
   PDF alone would silently skip the endpoint.

Use a single DB filename everywhere: **`ps06.db`**.

## Run procedure

### Step A — register the case/document (status package)
```bash
./venv311/bin/python -m ps06.status.cli --db ps06.db init-db
./venv311/bin/python -m ps06.status.cli --db ps06.db add-doc --case realcase1 --doc doc1
```

### Step B — normal extraction run (proves endpoint + VLM path)
Use a **fresh `--doc` id** for every run. Force the VLM path with a scanned doc, or add
`--no-text-layer-fast-path` on a text-layer doc:
```bash
./venv311/bin/python -m ps06.ocr.cli --db ps06.db run \
  --case realcase1 --doc doc1 --file /path/to/scanned_doc.pdf \
  --endpoint-url "$ENDPOINT_URL" --model-name "$MODEL_NAME" \
  --no-text-layer-fast-path \
  --auth fake --fake-token "$JWT" \
  --json
```
Confirm on the **first** VLM call that the endpoint accepts the request — the code always
sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on every call
(`http_client.py:97`); some serving runtimes reject unknown `extra_body`. If you see
`413`/timeouts, re-run with a lower `--pdf-dpi` (default 300).

### Step C — deliberate 401-recovery test (proves the re-mint/retry)
Seed a bad token first, the real token second. Construction mints the bad token → first
call 401s → wrapper re-mints and picks up `$JWT` → retry succeeds:
```bash
./venv311/bin/python -m ps06.ocr.cli --db ps06.db run \
  --case realcase1 --doc doc2 --file /path/to/scanned_doc.pdf \
  --endpoint-url "$ENDPOINT_URL" --model-name "$MODEL_NAME" \
  --no-text-layer-fast-path \
  --auth fake --fake-token "not-a-real-token" --fake-token "$JWT" \
  --json
```
Expected: run succeeds, doc ends `OCR_DONE`. **Key finding to record:** if instead it
fails without retrying, the endpoint returned something other than a `401`
(`openai.AuthenticationError`) for the bad token — e.g. `403`/redirect — which means the
production retry path would never fire. That is exactly the signal worth capturing now.

### Step D — inspect results (same DB)
```bash
./venv311/bin/python -m ps06.ocr.cli --db ps06.db show --case realcase1 --doc doc1 --json
./venv311/bin/python -m ps06.status.cli --db ps06.db show-case --case realcase1
```
Sanity-check `page_count`, `pages_via_text_layer` vs `pages_via_vlm`,
`orientation_corrections`, `acroform_fields`/`xfa_fields`, `processing_seconds`, and the
extracted text.

## Data-handling (real applicant data — mandatory)
- **Unencrypted at rest.** `ps06.db`'s `document_extraction.payload` stores full OCR text +
  form fields in plaintext. Treat the DB file with the same care as the source documents;
  keep it in a scoped, non-shared, uncommitted location and delete when done.
- **Logged output.** `--json` prints the full extracted text to session output, which
  Workbench logs. Drop `--json` (use `show` on demand) if session logs are retained.
- **Token in process list.** `"$JWT"` on the command line is visible via `ps` even though
  it stays out of shell history. Acceptable on a single-tenant session; avoid on shared hosts.

## What this validates / does not
**Validated:** endpoint reachability, VLM path, orientation correction, AcroForm/XFA
extraction, status-table writes, and the 401 re-mint/retry **recovery** path (Step C).
**Not validated:** real Knox token minting (`CDSWAuthProvider` still a stub), recovery
from a token that expires *naturally* mid-document, and anything prod-Workbench-specific.

## No code changes
Tier 1 is a runbook only — no edits to `ps06/`. Implementing a real
`CDSWAuthProvider.mint_token()` (likely reading `/tmp/jwt`) is separate M-3/open-item-#4
work with its own review, per the one-script-at-a-time working agreement.

## Verification of this plan
Each corrected step maps to verified source: fast-path (`extraction.py`), retry control
flow (`http_client.py:72-83`), token exhaustion (`auth.py:68-72`), re-run FAILED
(session-4 log). Success = Steps B and C both end `OCR_DONE`, `show` returns a coherent
payload, and the Step C behavior (retry vs no-retry) is recorded.
