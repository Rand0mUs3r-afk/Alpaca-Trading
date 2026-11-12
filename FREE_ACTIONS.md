# Free paper research with Gemini/Groq and quota handling

This is research only: no orders, cancellations, live trading, schedule or permanent
web dashboard. The existing five analysts and debates remain.

## Secrets and running

Repository Settings -> Secrets and variables -> Actions.
Required: ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_EXPECTED_PAPER_ACCOUNT_ID.
For analysis add SEC_USER_AGENT (your name and contact email) and at least one of:
- GOOGLE_API_KEY from a Google project with paid billing disabled.
- GROQ_API_KEY from the Groq Free plan.
Optional: FRED_API_KEY, FINNHUB_API_KEY. Only use permitted free-access endpoints.
No OpenAI key is needed or passed. The app cannot inspect your billing plan.

Actions -> Free AI - Paper research -> Run workflow -> main.
Choose check for account verification, or analyze with one stock such as AAPL.
Provider groq uses openai/gpt-oss-120b at Groq, not at OpenAI. Provider auto tries
configured Gemini then Groq; provider gemini uses Google only. The model dropdown
selects only the Gemini model and is ignored in Groq mode. Confirm free billing.
Read the Summary and sanitized artifact; public runs have public research results.

## Groq quota handling

Groq output is capped at 2048 tokens rather than 4096. Retrieved downstream report
context is reduced to 2400 tokens with at most 8 chunks and 6000 memory characters.
This keeps all five roles but can reduce report detail and change decisions. It is
not an equal-accuracy, equal-hit-rate or profitability guarantee.

Before a synchronous Groq chat request, the wrapper estimates input plus reserved
output plus 256 tokens of overhead. It uses a compatible local tiktoken encoding
if available, otherwise a conservative byte-length estimate. This is NOT an exact
Groq token count. A request above the configured 7500-token budget is stopped
before sending, without silently cutting messages or tool results.

A conservative rolling 65-second reservation window reduces token/minute bursts.
Observed server token limits can lower the local budget. Server remaining-token
and reset headers are honored. Other apps using the same organization can still
consume quota and cause 429s. Cached input can make this policy over-conservative.

For 429s, the wrapper reads numeric rate-limit/reset headers and classifies the
response without printing raw response text or organization IDs. It waits for
Retry-After/token reset when available, otherwise 65 seconds. At most three HTTP
attempts are allowed per synchronous quota-limited Groq call. A reset requiring
more than 120 seconds stops the request instead of sleeping indefinitely.

Known daily quota exhaustion and requests reported above the provider's limit stop
without futile retries. The result distinguishes GROQ_DAILY_QUOTA_EXHAUSTED,
GROQ_REQUEST_TOO_LARGE, GROQ_RESET_TOO_FAR_AWAY and minute/unknown limits. A local
preflight rejection is explicitly marked quota_source=local_preflight; its status
is not a claim that Groq returned that response. Unknown 429s can still exhaust the
bounded retries. Free quotas cannot be bypassed and are not guaranteed sufficient.

## Other behavior

The previous bounded 408/5xx retries and whole-analysis provider fallback remain.
Authentication, permission and coding failures do not silently switch providers.
The workflow timeout is 90 minutes to allow quota waits; Cancel workflow still stops
it. It runs only when manually started, on standard public GitHub-hosted runners.
No new secrets or paid fallback are introduced.

RSS headlines and SEC Company Facts still replace hosted search; unavailable social
or optional data remains a limitation. Local FastEmbed memory starts fresh each run.
Read-only broker guards remain. HTTP quotas are handled for the synchronous Groq
adapter used here, not as a general-purpose async/network sandbox.

Offline tests use fake clocks/responses and do not prove a complete live analysis.
Check the full GitHub test run and then one real model run. A request still too large
needs a deliberately smaller evidence profile or another suitable free provider;
repeatedly rerunning it will not solve a per-request budget failure.
