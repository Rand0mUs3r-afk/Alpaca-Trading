"""Quota-aware wrapper for the existing read-only Gemini/Groq runner."""
from collections import deque
import json
import math
import re
import threading
import time
from urllib.parse import urlparse

TOKEN_BUDGET = 7500
OUTPUT_TOKENS = 2048
SNAPSHOT = {}


class QuotaStop(RuntimeError):
    def __init__(self, code, details):
        super().__init__(code)
        self.details = {'error_code': code, **details}
        self.code = 413 if code == 'GROQ_REQUEST_TOO_LARGE' else 429


def seconds(value):
    value = str(value or '').strip()
    if re.fullmatch(r'\d+(?:\.\d+)?', value):
        number = float(value)
        return number if math.isfinite(number) else None
    pieces = re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h|d)', value)
    if not pieces or ''.join(number + unit for number, unit in pieces) != value:
        return None
    return sum(float(number) * {'ms': .001, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}[unit]
               for number, unit in pieces)


def safe_headers(headers):
    headers = {str(key).lower(): value for key, value in headers.items()}
    result = {}
    for key in ('limit-requests', 'limit-tokens', 'remaining-requests', 'remaining-tokens'):
        value = str(headers.get('x-ratelimit-' + key, ''))
        if re.fullmatch(r'\d{1,15}', value):
            result[key] = int(value)
    for key, source in (('retry_after_seconds', 'retry-after'),
                        ('reset_tokens_seconds', 'x-ratelimit-reset-tokens'),
                        ('reset_requests_seconds', 'x-ratelimit-reset-requests')):
        value = seconds(headers.get(source))
        if value is not None and 0 <= value <= 604800:
            result[key] = value
    return result


def quota_reason(response):
    try:
        message = str(response.json().get('error', {}).get('message', '')).lower()
    except Exception:
        message = ''
    info = safe_headers(response.headers)
    counts = {}
    for key, value in re.findall(r'\b(limit|used|requested)\s*[:=]?\s*([\d,]+)', message):
        counts[key] = int(value.replace(',', ''))
    if counts:
        info['reported_token_counts'] = counts
    if (response.status_code == 413 or 'request too large' in message or
            counts.get('requested', 0) > counts.get('limit', float('inf'))):
        reason = 'GROQ_REQUEST_TOO_LARGE'
    elif re.search(r'\b(tpd|rpd)\b|tokens per day|requests per day', message) or info.get('remaining-requests') == 0:
        reason = 'GROQ_DAILY_QUOTA_EXHAUSTED'
    else:
        reason = 'GROQ_MINUTE_OR_UNKNOWN_LIMIT'
    return reason, {'quota_source': 'groq_response', 'rate_limits': info}


def estimate_tokens(payload):
    text = json.dumps(payload, ensure_ascii=False)
    try:
        import tiktoken
        try:
            encoder = tiktoken.get_encoding('o200k_harmony')
        except Exception:
            encoder = tiktoken.get_encoding('o200k_base')
        input_tokens = len(encoder.encode_ordinary(text))
    except Exception:
        input_tokens = (len(text.encode('utf-8')) + 1) // 2
    output = max(int(payload.get('max_tokens') or 0),
                 int(payload.get('max_completion_tokens') or 0), OUTPUT_TOKENS)
    return input_tokens + output + 256


class Budget:
    def __init__(self, limit=TOKEN_BUDGET, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.clock, self.sleep = limit, clock, sleep
        self.events = deque()
        self.lock = threading.Lock()
        self.server = {}
        self.server_reset = 0

    def update(self, headers):
        info = safe_headers(headers)
        self.server = info
        if info.get('limit-tokens', 0) > 256:
            self.limit = min(self.limit, info['limit-tokens'] - 256)
        self.server_reset = self.clock() + info.get('reset_tokens_seconds', 0) + 1

    def reserve(self, needed):
        if needed > self.limit:
            raise QuotaStop('GROQ_REQUEST_TOO_LARGE', {
                'quota_source': 'local_preflight', 'estimated_request_tokens': needed,
                'configured_token_budget': self.limit,
                'quota_help': 'Request will not fit this budget. Waiting alone cannot fix it. No evidence was silently cut.'})
        with self.lock:
            while True:
                now = self.clock()
                while self.events and now - self.events[0][0] >= 65:
                    self.events.popleft()
                if self.server.get('remaining-requests') == 0:
                    raise QuotaStop('GROQ_DAILY_QUOTA_EXHAUSTED', {
                        'quota_source': 'groq_headers', 'rate_limits': self.server})
                wait = 0
                if sum(count for _, count in self.events) + needed > self.limit:
                    wait = self.events[0][0] + 65 - now
                if self.server.get('remaining-tokens', needed) < needed and now < self.server_reset:
                    wait = max(wait, self.server_reset - now)
                if wait <= 0:
                    self.events.append((now, needed))
                    return
                print('GROQ_BUDGET_WAIT_SECONDS: ' + str(round(wait, 2)))
                self.sleep(wait)


def send_with_limits(send, client, request, args, kwargs, budget, sleep=time.sleep, estimate=estimate_tokens):
    try:
        payload = json.loads(request.content)
        needed = estimate(payload)
    except (ValueError, TypeError, AttributeError):
        raise QuotaStop('GROQ_REQUEST_SIZE_UNKNOWN', {'quota_source': 'local_preflight'})
    for attempt in range(3):
        budget.reserve(needed)
        response = send(client, request, *args, **kwargs)
        budget.update(response.headers)
        if response.status_code not in (413, 429):
            return response
        reason, info = quota_reason(response)
        SNAPSHOT.clear()
        SNAPSHOT.update(error_code=reason, **info)
        if reason in ('GROQ_REQUEST_TOO_LARGE', 'GROQ_DAILY_QUOTA_EXHAUSTED'):
            response.close()
            raise QuotaStop(reason, info)
        if attempt == 2:
            return response
        limits = info['rate_limits']
        delay = max(limits.get('retry_after_seconds', 0), limits.get('reset_tokens_seconds', 0))
        delay = delay + 1 if delay else 65
        if delay > 120:
            response.close()
            raise QuotaStop('GROQ_RESET_TOO_FAR_AWAY', info)
        response.close()
        print('GROQ_RATE_WAIT_SECONDS: ' + str(round(delay, 2)))
        sleep(delay)


def install(free):
    SNAPSHOT.clear()
    original_guard = free.install_http_guard
    def guarded_install():
        original_guard()
        import httpx
        original_send = httpx.Client.send
        budget = Budget()
        def send(self, request, *args, **kwargs):
            url = urlparse(str(request.url))
            if url.hostname == 'api.groq.com' and url.path.rstrip('/') == '/openai/v1/chat/completions':
                return send_with_limits(original_send, self, request, args, kwargs, budget)
            return original_send(self, request, *args, **kwargs)
        httpx.Client.send = send
    free.install_http_guard = guarded_install
    original_builder = free.build_llm
    def build(provider, *args, **kwargs):
        llm = original_builder(provider, *args, **kwargs)
        if provider == 'groq':
            llm.max_tokens = OUTPUT_TOKENS
        return llm
    free.build_llm = build
    original_graph = free.run_graph
    def run(provider, model, graph_class):
        if provider != 'groq':
            return original_graph(provider, model, graph_class)
        def compact_graph(config):
            config = dict(config)
            config.update(report_context_budget_tokens=2400, report_context_max_chunks=8,
                          report_context_memory_chars=6000)
            return graph_class(config=config)
        return original_graph(provider, model, compact_graph)
    free.run_graph = run
    original_details = free.error_details
    def details(exc):
        info = original_details(exc)
        if isinstance(exc, QuotaStop):
            info.update(exc.details)
        elif free.http_status(exc) in (413, 429) and SNAPSHOT:
            info.update(SNAPSHOT)
        return info
    free.error_details = details


def main():
    from scripts import free_providers
    install(free_providers)
    return free_providers.main()


if __name__ == '__main__':
    raise SystemExit(main())
