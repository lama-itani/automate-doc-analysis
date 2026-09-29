# PS-06 Tier-1 Workbench Test — Results (2026-09-28)

**Verdict:** Endpoint, VLM path, status table and 401 retry all work live. One blocking defect: orientation correction can corrupt a correct image and produce wrong data that is marked `OCR_DONE`.

## Environment
- Workbench Python is 3.10 (no 3.11). `pyproject.toml` allows `>=3.10`. Used `venv310`.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`. Endpoint accepted `extra_body` (`enable_thinking: False`).

## Results
| Phase / step | Input | Outcome |
|---|---|---|
| 0: unit tests | — | 81 passed |
| 1: CLI smoke | synthetic text-layer PDF | `OCR_DONE`, no network call |
| B: normal run | real ID photo, 1 VLM page | `OCR_DONE`, 14.6 s, **wrong text** (see defect) |
| C: 401 recovery | synthetic scan, bad → real token | `OCR_DONE`, exact OCR |
| E1: retry exhausted | two bad tokens | `FAILED`, `AuthRetryExhausted` |
| E2: tokens exhausted | one bad token | `FAILED`, `RuntimeError` |
| doc5: multi-page | real scan, 2 VLM pages | `OCR_DONE`, 78.1 s |
| doc6: ID recto-verso | real scan, 2 VLM pages, both upright | `OCR_DONE`; page 1 `NONE` (correct), page 2 `FLIP_VERTICAL` (**false positive**) |

**Key signal:** the endpoint returns a genuine `401` for a bad token, so the production re-mint/retry path fires.

## Findings
1. **Blocking: orientation false positive corrupts data silently.** An upright photo of a person holding a passport was labelled `FLIP_VERTICAL`. `_correct_orientation` applies `ImageOps.flip` (a mirror), so the text became mirrored and the OCR output did not match the passport. The doc still ended `OCR_DONE`. Causes:
   - The prompt describes `FLIP_VERTICAL` as "upside-down", which overlaps with `ROTATE_180`. A true upside-down image needs a rotation, not a mirror.
   - Real scans and photos are almost never mirrored, so the `FLIP_*` labels add risk and no value.
   - There are no 90°/270° labels, so sideways photos are not handled.
   - The label is applied with no check that the correction improved the result.
   - **Reproduced on doc6:** the upright back of an ID card (MRZ `<<<` lines + QR code, little plain text) was also labelled `FLIP_VERTICAL`. Pages with sparse text or MRZ chevrons are a likely trigger. 2 false positives out of 5 real VLM pages.
2. **Model output wraps text in `` ```markdown `` fences.** Nothing strips it; downstream rules will see it.
3. **Timing:** 12–39 s per VLM page. A 3–5 document batch fits well inside a ~1 h JWT.
4. **Fixture note:** the fast path needs ≥60 chars of text per page (`min_text_layer_chars`).

## Suggested fix (separate work, own review)
- Replace labels with rotations only: `0 / 90 / 180 / 270`. Drop the mirror flips.
- Add a guard: if a correction is applied, fall back to the uncorrected image when the corrected OCR is worse, or flag the doc for review instead of `OCR_DONE`.
- Add regression tests with an upright photo and an upright ID back side (MRZ + QR); both must return `NONE`.
- Consider validating ID reads with MRZ check digits: a deterministic accuracy check that needs no ground truth.
- Strip code fences from VLM output.

## Next step
Implement the orientation fix as its own reviewed change. Then rerun on both failing files with fresh `--doc` ids: `docs/985356/DocumentoIdentidad_RMS.pdf` and `docs/985356/CEDULAJOSE.pdf`. Pass: no orientation corrections, and extracted text that matches the documents.

## Not validated
Real Knox minting (`CDSWAuthProvider` is still a stub) and natural token expiry mid-document.
