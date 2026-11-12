# Free AI paper research: Gemini with Groq backup

This manual Actions workflow checks your Alpaca PAPER account and runs the existing
five analysts, debates and risk review. It DOES NOT submit, cancel or replace orders.
There is no schedule or permanently hosted dashboard.

## Actions secrets

Settings -> Secrets and variables -> Actions -> New repository secret.

Required for account check:
- ALPACA_API_KEY: paper key.
- ALPACA_SECRET_KEY: matching paper secret.
- ALPACA_EXPECTED_PAPER_ACCOUNT_ID: account ID, not API key or account number.

For analysis:
- GROQ_API_KEY: your Groq FREE-plan API key. Required for the Groq route/backup.
- GOOGLE_API_KEY: your Gemini key on an unbilled free-tier project. Optional if using Groq only.
- SEC_USER_AGENT: your name and contact email, for SEC filings.
- FRED_API_KEY and FINNHUB_API_KEY: optional data keys, using only your permitted access.

Do not add an OpenAI key. The workflow passes no OpenAI/Anthropic/OpenRouter key.
It cannot verify your provider billing plans. Confirm the supplied keys are free-plan
keys with paid billing disabled; a model name alone does not prevent charges on a
paid account. Free quotas can stop the run. No paid fallback is configured.

## Run from the browser

Actions -> Free AI - Paper research -> Run workflow -> branch main.

1. Choose check for an account check (no model request).
2. Choose analyze, ticker AAPL, and provider groq to test Groq directly.
   Groq uses openai/gpt-oss-120b through Groq, not through an OpenAI account.
3. Choose provider auto for Gemini first, then Groq if Gemini is unavailable.
4. The Gemini model input only selects the Google model. It is ignored by Groq.
5. Tick the free-plan confirmation only after checking your provider accounts.
6. Read Summary and the sanitized result artifact. provider/model show the actual
   successful provider; provider_attempts records failures and HTTP codes.

If you create a run using groq only, no Google key is required. In auto mode only
providers with configured keys are attempted. An absent backup key is not invented.

## Retry and fallback rules

Each model-generation call has at most two wrapper attempts for transient 408/5xx
failures, with a 20-second wait. Gemini SDK retry setting is 1; Groq SDK retries are 0.
The SDK may therefore perform additional bounded HTTP requests inside a wrapper
attempt. A 429 does not loop on the same provider. After a provider failure, auto
may restart the ENTIRE analysis with the next provider for 404/408/429/selected 5xx
or recognized connection timeouts. It does not switch on coding errors, bad-key
401s, permission 403s or malformed-request 400s. No in-progress tool conversation
or provider-specific thought signatures are passed between providers.

The graph and all five analyst roles remain. Synchronous calls are paced and
analysts run sequentially. This is not an equal-accuracy or profitability claim.
Free Groq token limits may be smaller than the assembled research prompt or total
run; a provider can therefore still fail even though its account is free.

## Data, memory, privacy and limits

Existing free RSS headline retrieval and SEC Company Facts replace hosted OpenAI
search. This is not full-article or complete social-data coverage. Local FastEmbed
embeddings are used. Memory starts fresh each Actions run, with separate stores
for provider attempts. All current broker operations are read-only.

The exact approved Groq chat-completions endpoint is allowed; known unapproved AI
endpoints and Alpaca writes/live trading are blocked through Requests/HTTPX. This
is defense in depth, not an operating-system sandbox. Public Actions logs and
research summaries are visible. No raw graph/memory databases are uploaded.

Standard public GitHub-hosted runners are used. Keep within GitHub/provider policies.
Unit tests use mocks; they do not certify a successful live Gemini/Groq integration
or successful execution of the complete analyst graph. Automatic paper orders
remain a separate, unimplemented feature pending end-to-end verification.
