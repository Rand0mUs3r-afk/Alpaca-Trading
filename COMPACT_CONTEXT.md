# Groq-only compact source prompts

The workflow starts scripts.free_compact_context. It installs the existing request
fitting, quota guards and Gemini/Groq router. Orders remain disabled.

The original debate prompts request full untruncated reports PLUS a scored claim
matrix, a digest and full debate history. Reducing tool descriptions cannot fix a
single large no-tools prompt. The Groq profile stops resending full reports when a
claim matrix is present, and omits full-history placeholders when a digest exists.
Latest counterarguments, role instructions and decision-format placeholders stay.
Original templates and full state/reports are not overwritten or deleted.

At runtime, a temporary override directory is populated from the active templates.
Groq uses these overrides for propagate(). The prior prompt-directory setting and
context defaults are restored even on failure. Gemini uses the original profile.

The profile uses 1600 evidence-budget tokens, at most six selected chunks and four
points/claims per report. Compact claims/excerpts are shorter, and the recent digest
is configured for four messages / 1800 characters. Calls that omit an explicit
context config receive these same defaults while the Groq graph is running.
All five analyst roles and the existing debate/risk round counts are retained.

This is SELECTED EVIDENCE, not all evidence in every request. Some observations and
older arguments may be omitted; conclusions and target hit rates are not guaranteed
unchanged. Models are instructed to prefer HOLD/NEUTRAL if evidence is insufficient.
No claim is made that the profile guarantees success or profitable trading.

The local request-size check still runs. It is not raised or bypassed. If a request
still cannot fit, its numeric report identifies that failure. The profile primarily
addresses templates containing both full reports and claim matrices, and templates
with a debate digest; other oversized prompt types may still require attention.
Offline tests verify prompt substitution, role preservation and restoration, not
full real-model analysis. No new keys, paid service or order execution is added.
