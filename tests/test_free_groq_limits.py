"""Offline quota tests; no provider keys, models or broker calls."""
import json
import unittest
from types import SimpleNamespace
from scripts import free_groq_limits as quota


class FakeTime:
    def __init__(self):
        self.now = 0
        self.waits = []
    def clock(self):
        return self.now
    def sleep(self, value):
        self.waits.append(value)
        self.now += value


class FakeResponse:
    def __init__(self, code=200, message='', headers=None):
        self.status_code, self.message = code, message
        self.headers = headers or {}
        self.closed = False
    def json(self):
        return {'error': {'message': self.message}}
    def close(self):
        self.closed = True


class QuotaTests(unittest.TestCase):
    def setUp(self):
        quota.SNAPSHOT.clear()

    def test_duration_parsing(self):
        self.assertAlmostEqual(quota.seconds('1m2.5s'), 62.5)
        self.assertAlmostEqual(quota.seconds('500ms'), .5)
        self.assertEqual(quota.seconds('2'), 2)
        self.assertIsNone(quota.seconds('Bearer private-secret'))

    def test_headers_exclude_sensitive_values(self):
        info = quota.safe_headers({'x-ratelimit-limit-tokens': '8000', 'Authorization': 'secret',
                                   'retry-after': '2', 'x-ratelimit-reset-tokens': '7.66s'})
        self.assertEqual(info['limit-tokens'], 8000)
        self.assertEqual(info['reset_tokens_seconds'], 7.66)
        self.assertNotIn('secret', json.dumps(info))

    def test_daily_limit_is_not_treated_as_minute_limit(self):
        reason, info = quota.quota_reason(FakeResponse(429, 'Tokens per day (TPD) reached in private-org'))
        self.assertEqual(reason, 'GROQ_DAILY_QUOTA_EXHAUSTED')
        self.assertNotIn('private-org', json.dumps(info))

    def test_413_without_message_is_not_retried(self):
        reason, info = quota.quota_reason(FakeResponse(413))
        self.assertEqual(reason, 'GROQ_REQUEST_TOO_LARGE')

    def test_requested_above_limit_is_too_large(self):
        reason, info = quota.quota_reason(FakeResponse(429, 'Limit 8000 Used 0 Requested 12000 private-org'))
        self.assertEqual(reason, 'GROQ_REQUEST_TOO_LARGE')
        self.assertEqual(info['rate_limits']['reported_token_counts']['requested'], 12000)

    def test_oversized_request_stops_before_send(self):
        budget = quota.Budget(limit=7500)
        with self.assertRaises(quota.QuotaStop) as caught:
            budget.reserve(8000)
        self.assertEqual(caught.exception.details['quota_source'], 'local_preflight')

    def test_rolling_budget_waits_without_network(self):
        clock = FakeTime()
        budget = quota.Budget(limit=7500, clock=clock.clock, sleep=clock.sleep)
        budget.reserve(5000)
        budget.reserve(4000)
        self.assertEqual(clock.waits, [65])

    def test_server_token_reset_is_honored(self):
        clock = FakeTime()
        budget = quota.Budget(clock=clock.clock, sleep=clock.sleep)
        budget.update({'x-ratelimit-remaining-tokens': '10', 'x-ratelimit-reset-tokens': '4s'})
        budget.reserve(100)
        self.assertEqual(clock.waits, [5])

    def test_minute_429_waits_then_succeeds(self):
        clock = FakeTime()
        budget = quota.Budget(clock=clock.clock, sleep=clock.sleep)
        responses = [FakeResponse(429, 'TPM limit reached', {'retry-after': '2'}), FakeResponse()]
        calls = []
        def send(*args, **kwargs):
            calls.append(1)
            return responses.pop(0)
        request = SimpleNamespace(content=b'{}')
        response = quota.send_with_limits(send, None, request, (), {}, budget, sleep=clock.sleep, estimate=lambda payload: 100)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 2)
        self.assertEqual(clock.waits, [3])

    def test_daily_429_does_not_retry(self):
        clock = FakeTime()
        budget = quota.Budget(clock=clock.clock, sleep=clock.sleep)
        calls = []
        def send(*args, **kwargs):
            calls.append(1)
            return FakeResponse(429, 'TPD exhausted')
        with self.assertRaises(quota.QuotaStop):
            quota.send_with_limits(send, None, SimpleNamespace(content=b'{}'), (), {}, budget, sleep=clock.sleep, estimate=lambda payload: 100)
        self.assertEqual(len(calls), 1)
        self.assertEqual(clock.waits, [])

    def test_unknown_429_has_bounded_retries(self):
        clock = FakeTime()
        budget = quota.Budget(clock=clock.clock, sleep=clock.sleep)
        calls = []
        def send(*args, **kwargs):
            calls.append(1)
            return FakeResponse(429)
        response = quota.send_with_limits(send, None, SimpleNamespace(content=b'{}'), (), {}, budget, sleep=clock.sleep, estimate=lambda payload: 100)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(calls), 3)
        self.assertEqual(clock.waits, [65, 65])

    def test_profile_changes_only_groq_and_preserves_agent_selection(self):
        original_config = {'selected_analysts': ['market', 'social', 'news', 'fundamentals', 'macro']}
        seen = []
        free = SimpleNamespace(
            install_http_guard=lambda: None,
            build_llm=lambda *args, **kwargs: SimpleNamespace(max_tokens=4096),
            run_graph=lambda provider, model, cls: cls(config=original_config),
            error_details=lambda exc: {}, http_status=lambda exc: None)
        quota.install(free)
        groq_llm = free.build_llm('groq')
        google_llm = free.build_llm('google')
        self.assertEqual(groq_llm.max_tokens, 2048)
        self.assertEqual(google_llm.max_tokens, 4096)
        free.run_graph('groq', 'model', lambda config: seen.append(config))
        self.assertEqual(seen[0]['report_context_budget_tokens'], 2400)
        self.assertEqual(seen[0]['selected_analysts'], original_config['selected_analysts'])
        self.assertNotIn('report_context_budget_tokens', original_config)


if __name__ == '__main__':
    unittest.main()
