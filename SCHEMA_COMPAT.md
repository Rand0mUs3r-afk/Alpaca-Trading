# Groq structured schema and final-manager context compatibility

The Actions launcher is scripts.free_schema_compat. It installs the existing compact
context, request fitting and quota policies, then adjusts only Groq response-format
schemas before they are sent. Original Pydantic model definitions and local response
validation are not changed. No keys, paid fallback or order permissions are added.

The reported failure is an anyOf enum/null representation for advisory_rating.
The wire transform rewrites a narrowly supported nullable primitive union as a type
array, adding null to its enum when needed. It handles local enum references and
plain nullable strings/numbers/booleans. Other unions, external references, parent
validation constraints and non-scalar object unions are left unchanged.

Response-format name, strict flag, required fields, additionalProperties and
non-nullable action enums stay. The model may still fail to produce a valid output;
the existing Pydantic validation and application fallback behavior remain active.
Standards-equivalent schemas do not prove acceptance by every Groq server version.
A real run is still required to verify remote acceptance.

The previous prompt profile detected debate_digest but not risk_debate_digest.
The final manager therefore could still receive full risk history along with its
digest. The added transform handles manager-specific digest placeholders, retaining
account state, trader plan, digest, role instructions and final decision format.
It does not delete full history from the stored graph state.

Logs show GROQ_NULLABLE_SCHEMA_FIELDS when fields are normalized. Existing numeric
request-size and quota diagnostics remain. Full reports use selected-evidence
prompts as documented in COMPACT_CONTEXT.md, so analysis may differ from upstream.

The 400 messages in the supplied log were caught and did not alone prove the final
run failure. If a subsequent run fails, inspect its Summary JSON for the final cause.
Offline tests check transformations, not profitable trading or full live operation.
