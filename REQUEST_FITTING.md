# Groq request fitting and wrapped local errors

The workflow now starts scripts.free_request_fit, which installs the existing
quota-aware runner and adds request fitting. No new keys or order mode are added.

If the Groq request estimate is too high, long tool-help descriptions are shortened
to 180 characters. Tool names, parameter schemas, required fields and the messages
containing evidence are retained. Shorter help can still affect tool-use quality.

The wrapper then fits the output allowance into the existing token budget, with a
64-token extra margin. It keeps up to 2048 output tokens and requires at least 512,
unless the original requested allowance was smaller. It does not delete financial
observations, conversation messages or tool results to force an input to fit.

If input still cannot fit, it stops and reports estimated input/overhead, total,
budget and message/tool counts. These are conservative estimates, not Groq's exact
billing counts. It does not raise the provider allowance or guarantee completion.

The SDK may wrap an exception from our local guard as a connection error. The new
reporter searches the original exception chain and recovers the QuotaStop details.
A locally generated stop is labeled status_is_local=true with local_status; the
http_status field is null rather than pretending Groq returned that HTTP response.
Remote quota responses remain marked groq_response.

Only request-size numbers and counts are printed. Headers, keys, raw evidence and
raw exception messages are not added to the diagnostic output. When output is
changed, the HTTP request is rebuilt with its original URL, auth headers and
extensions, and a recalculated content length. Existing read-only guards remain.

Unit tests cover fitting and wrapped exceptions, not successful real model trading.
The complete workflow and full model analysis still need verification. All five
agents remain, but answer limits and shorter tool help can alter analysis quality.
If the input remains too large, waiting or rerunning unchanged will not make it fit.
