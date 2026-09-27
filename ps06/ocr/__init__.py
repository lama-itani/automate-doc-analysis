"""Per-document OCR+Rules job — M-2.

Each invocation processes exactly one document: mints its own fresh auth token,
performs OCR/VLM extraction, writes the result and a status transition, and
exits. See the M-2 plan for design; rules/aggregation across a case's documents
is a later milestone (M-4), not part of this package.
"""
