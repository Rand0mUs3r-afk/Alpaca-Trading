# Free AI paper research from the Actions tab

This is a manual research workflow, not a hosted dashboard and not an automatic
trading deployment. It reads an Alpaca paper account and runs the existing five
analysts, research debate and risk review. It never submits, cancels or replaces
orders. Existing application defaults and workflows are not changed.

## Add Actions secrets (not Codespaces secrets)

Repository -> Settings -> Secrets and variables -> Actions -> New repository secret.

Required for `check`:
- `ALPACA_API_KEY`: paper-account API key.
- `ALPACA_SECRET_KEY`: matching paper-account secret.
- `ALPACA_EXPECTED_PAPER_ACCOUNT_ID`: the account ID displayed for that paper account.
  This is the broker's account ID, not the API key, account nickname or account number.

Also required for `analyze`:
- `GOOGLE_API_KEY`: Gemini Developer API key from a project with billing disabled.
- `SEC_USER_AGENT`: your name and contact email, such as `Your Name you@example.com`.

Optional free-access data keys:
- `FRED_API_KEY`: gives the macro analyst economic data.
- `FINNHUB_API_KEY`: uses endpoints available to your account. A free key does not
  guarantee entitlement to all insider/news endpoints.

Do not add an OpenAI key to this workflow. Do not commit .env files or credentials.
The workflow deliberately does not pass any paid-provider secrets to its process.

## Click-to-run

1. Actions -> Free AI - Paper research -> Run workflow.
2. Choose `check` first. The account must match the expected ID and be active.
3. After that succeeds, choose `analyze` and one US stock, such as AAPL.
4. Choose `gemini-3.8-flash` or `gemini-3.6-flash`.
5. Confirm your Google project has billing disabled. The app cannot inspect your
   Google billing plan; the same model can cost money on a paid project.
6. Open the run's Summary for the result. A sanitized JSON artifact is also saved.

This public repository has public Actions logs and summaries. Research decisions
are visible. No account balance, holdings list or account ID is intentionally
published. Do not enter personal/private information into the stock-symbol field.

## What stays and what changes

The existing graph and all five analyst roles stay. Defaults for research and risk
debate rounds stay. Calls are paced and analysts run sequentially to reduce quota
pressure. This may take many minutes. If free quota/model access is insufficient,
the run fails rather than switching to a paid model. Use GitHub's Cancel workflow
button to stop it.

The Actions runner replaces legacy OpenAI news-search tools with Google News RSS
headline evidence, and its fundamentals-search tool with SEC US-GAAP Company Facts.
The legacy tool names remain for compatibility; those replacements do not call
OpenAI. RSS is not full-article research or social sentiment. Missing optional data
may make reports incomplete. This is not an equal-accuracy or profitability claim.

Memory embeddings use the local open-source BAAI/bge-small-en-v1.5 model through
FastEmbed; its weights download on the GitHub runner. No embedding API is billed.
Memory starts fresh on every Actions run: cross-run learning is NOT implemented
in this first version. Raw logs, memory databases and graph states are not uploaded.

HTTP guards block known paid-provider endpoints, live Alpaca API access and Alpaca
writes through Requests/HTTPX. The runner does not expose an order-execution mode.
This is defense in depth, not a general-purpose operating-system sandbox.

## Costs and limits

Use only standard GitHub-hosted runners in a public repository and remain within
GitHub policies. Free-model quotas and availability change. Keeping the Google
project unbilled is required for the no-payment goal. No Google Search grounding,
paid model fallback, public dashboard tunnel or permanent background job is used.
There is no scheduled trigger. Configure/test automatic paper execution separately
only after account checks, data quality and end-to-end tests succeed.

Local unit tests cover input validation and redaction. They do not prove live model
compatibility, the availability of SEC/RSS data, full graph completion or successful
broker fills. No remote integration run has been performed merely by adding files.
