# automate-doc-analysis (PS-06)

Checks immigration case files. You upload a case's documents. The app reads each document, classifies it, applies business rules, and returns a verdict: **VERDE**, **AMARILLO** or **ROJO**.

It runs on Cloudera AI as one Application (API + web UI). The Application starts one Cloudera AI Job per document.

## How it works

1. **Upload.** The user uploads a case in the UI. The API saves the files to `PS06_DATA_DIR/uploads/<case>/` and records each document as `RECEIVED` in a SQLite status table.
2. **One job per document.** A background thread starts one Cloudera AI Job per document (script `jobs/ocr_job.py`), up to 3 at a time. Each job reads its own fresh token from `/tmp/jwt`. This avoids token expiry on long runs.
3. **Read and classify.** The job reads the text layer of each PDF page. Images, and pages with under 60 characters of text, go to the vision model (`PS06_ENDPOINT_URL`). The model then classifies the document and extracts fields.
4. **Report back.** The job writes a result file to `PS06_DATA_DIR/spool/<case>/<doc>.json`. The Application records it as `OCR_DONE` or `FAILED`. Only the Application writes the status DB (the project volume is NFS).
5. **Rules.** When every document is `OCR_DONE`, the rules engine runs. It is deterministic, no LLM. It checks the 5 expected documents, family lineage (applicant, parent, grandparent) and conflicts. Rules are in `ps06/rules/rules.yaml`.
6. **Verdict.** The UI shows the verdict, the issues, the document checklist and the lineage.

If any document fails, there is no verdict. The UI lists the failed documents and their errors.

### Code map

| Path | Role |
|---|---|
| `apps/api_app.py` | Application entry point |
| `jobs/ocr_job.py` | Job entry point (one document) |
| `ps06/api/` | FastAPI service: upload, status, verdict, serves the UI |
| `ps06/orchestrator/` | Launches and tracks one job per document |
| `ps06/ocr/` | Text and vision extraction, token handling, result files |
| `ps06/classification/` | Document type and field extraction (LLM) |
| `ps06/rules/` | Rules engine and `rules.yaml` |
| `ps06/status/` | SQLite status table |
| `ui/` | React UI. `ui/dist/` is the built version and is committed |
| `tests/` | Tests (`pytest`) |

### API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Status, model name, whether the UI is built |
| `GET /api/cases` | All cases |
| `POST /api/cases` | Upload a case (form fields `case_id`, `files`) |
| `GET /api/cases/{id}` | Stage and error of each document |
| `GET /api/cases/{id}/verdict` | `pending`, `failed` or `verdict` |

## Deploy to a Cloudera AI environment

### 1. Get the code into the project

The project files must match `main`. For example, in a session terminal: `git pull`.

If you changed `ui/src/`, rebuild the UI on your machine first, then commit `ui/dist/`:

```
cd ui && npm install && npm run build
```

### 2. Install the Python packages

Open a session with the runtime below (PBJ JupyterLab, Python 3.10). Run from the repo root:

```
python -m pip install --user -r requirements.txt
```

The packages go to `/home/cdsw/.local`. The Application and the jobs share it, so install once per project.

`requirements.txt` pins the versions tested live. Do not change them without a live test.

### 3. Create the Application

| Setting | Value |
|---|---|
| Script | `apps/api_app.py` |
| Runtime | PBJ JupyterLab, Python 3.10, Standard, 2026.08 |
| Env var (required) | `PS06_ENDPOINT_URL`: the model endpoint's OpenAI base URL, ending in `/v1` |

Do not set `CDSW_APP_PORT`; the platform sets it.

Optional env vars:

| Variable | Default |
|---|---|
| `PS06_MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `PS06_DATA_DIR` | `/home/cdsw/ps06_data` |
| `PS06_DB_PATH` | `<PS06_DATA_DIR>/ps06.db` |
| `PS06_MAX_CONCURRENCY` | `3` (jobs at once) |
| `PS06_UI_DIST` | `<repo>/ui/dist` |
| `PS06_JOB_RUNTIME` | `docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-standard:2026.08.1-b5` |
| `PS06_JOB_CPU` | `1` |
| `PS06_JOB_MEMORY_GB` | `4` |
| `PS06_JOB_TIMEOUT_SECONDS` | `1800` |

You do not create jobs by hand. The Application creates one per document through the Workbench API, with the `PS06_JOB_*` settings. It deletes each job after a success and keeps it after a failure, for debugging.

**New environment?** Check that the job runtime image exists there. If not, set `PS06_JOB_RUNTIME` to one that does (PBJ JupyterLab, Python 3.10).

**Endpoint with an internal certificate?** Set `SSL_CERT_FILE` as a **project** env var. Jobs do not inherit the Application's env vars.

### 4. Check it started

Open `<app URL>/api/health`. Expect `"status": "ok"` and `"ui_built": true`.

The first lines of the Application log show the endpoint, model, job runtime and job resources in use. Check them.

If it fails to start, the log names the cause: a missing or invalid setting, missing `CDSW_PROJECT_ID`, `cmlapi` not available, or the port in use.

### 5. Test with one document

Upload one small document as a test case. Check it reaches **Read complete** (`OCR_DONE`) before running real cases.

## Upload a case

1. Open the Application URL.
2. In **Upload case file**, enter the **Case number**. It must be new.
3. Drop the case's files, or click to browse.
4. Click **Process case file**.
5. Watch each document's stage. The page refreshes every 5 seconds.
6. When all documents finish, the verdict appears. Tick **Technical details** for per-document data.

File rules:

- Types: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.
- Max 50 MB per file. No empty files.
- The file name without extension is the document ID. Two files cannot share it (`a.pdf` and `a.png`).
- No names starting with `.`, no `/` or `\`, max 200 characters.

To run the same documents again, use a new case number.

If the Application restarts during a run, unfinished documents show **Interrupted**. They do not resume. Upload them again under a new case number.

## Data and cleanup

All case data is in `PS06_DATA_DIR`: `ps06.db` (plus `-wal`, `-shm`), `uploads/`, `spool/`. It holds applicant data in plain text. Delete it when you no longer need it.

## Run the tests

```
python -m pip install -e ".[dev,api]"
python -m pytest -q
```