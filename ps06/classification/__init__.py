"""Per-document classification (M-2.5).

Classifies a document's OCR-extracted text into one of a small set of
document types relevant to an immigration case file. Runs as a step inside
the existing OCR job (:mod:`ps06.ocr.job`), not a separate stage/invocation —
see :mod:`ps06.classification.classifier` for the rationale.
"""
