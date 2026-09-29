# PS-06 Tier-2 Classification Test — Results (2026-09-29)

**Verdict:** Classification works live. 4 out of 4 documents got the correct type on a real case folder, on both OCR paths. Scope is one folder, so this is a first positive signal, not proof of reliability.

## Setup
- Case folder `985356`, 4 PDFs, run as one case with `ps06.orchestrator.cli run-case`.
- Model `Qwen/Qwen2.5-VL-7B-Instruct`, `--auth fake` with a real JWT, `--max-concurrency 2`, orientation correction off.
- Environment fix: the Workbench checkout was older than the orchestrator merge (PR #8). `git pull` fixed `ModuleNotFoundError: ps06.orchestrator`. Use `venv310`, not `venv311`.

## Results
| Doc | Content | OCR path | Time | Type | Raw reply | Correct |
|---|---|---|---|---|---|---|
| `bc_parent` | Handwritten parent birth certificate | VLM, 2 pages | 75.5 s | `BIRTH_CERT` | `BIRTH_CERT` | ✅ |
| `app` | Anexo IV (LMD), fillable form | Text layer + form fields, 2 pages | 5.3 s | `APPLICATION` | `APPLICATION` | ✅ |
| `bc_applicant` | Applicant birth certificate (docx → PDF) | Text layer, 1 page | 6.2 s | `BIRTH_CERT` | `BIRTH_CERT` | ✅ |
| `id` | Photo of person holding passport | VLM, 1 page | 21.2 s | `ID_DOCUMENT` | `ID_DOCUMENT` | ✅ |

- All documents reached `OCR_DONE` on the first attempt. This was also the first live run of the M-3 orchestrator.
- `generation` was `None` for both birth certificates. That is expected, because neither states G1/G2/G3. Deciding generation is M-4's job.
- The model replied with clean labels only: no fences and no extra text. The parser's `OTHER` fallback was never needed.
- One classification call costs about 5 s. That is the time of the text-layer docs, which make no VLM call.

## Findings
1. **Handwriting is fine for classification.** The handwritten civil-registry certificate was read clearly enough to classify.
2. **The passport OCR is no longer mirrored**, because orientation correction was off. The text is readable, but it has two quality problems:
   - The model invented an image link (`![](https://example.com/image.png)`).
   - Some fields look merged or mislabelled. For example, the nationality seems to be joined to the surname. Field-level accuracy is not validated yet.
3. **Code fences are still in the VLM output** (Tier-1 finding 2, still open). They did not affect classification, but downstream rules will see them.
4. **The text layer contains ligature characters.** For example, `Certiﬁcado` uses the single `ﬁ` character. Rules that match text exactly could miss these, so normalize the text (NFKC) before rule matching.

## Not validated
- The `OTHER` and `EMPTY` types (no blank or unrelated page in the folder).
- Reliability across many folders. This was one folder with 4 documents.
- Field-level OCR accuracy.
- Real Knox token minting (Open Item #4).

## Suggested follow-ups (not blocking the build)
- Rerun on 2–3 more folders, including one blank page and one unrelated document.
- Strip code fences and invented image links from VLM output.
- Normalize ligatures in extracted text.
- Check ID reads with MRZ check digits (Tier-1 suggestion).

## Cleanup
Delete `classify_test.db`, `classify_test.db-wal` and `classify_test.db-shm`. They hold applicant text.
