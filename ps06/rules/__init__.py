"""M-4 rules / aggregation engine (Lineage + Semaphore).

A deterministic, non-LLM, case-level stage. Once every document in a case has
reached ``OCR_DONE`` it reads the persisted per-document extractions, resolves
the applicant's family lineage (G1/G2/G3), evaluates a VERDE/AMARILLO/ROJO
verdict, and writes a Spanish S0-S3 determination snapshot.

The whole package is pure logic over already-persisted data plus decoupled YAML
rule definitions — it touches no LLM and no Knox token, so it is fully testable
without live credentials (unlike the OCR/orchestrator layers). The business
logic ported here originally lived in the AEAD prototype's "Data Aggregation
Agent" ``goal`` prompt text (``workflow_template.json``); M-4 re-expresses it as
code so it is auditable and changeable without touching a prompt.
"""
