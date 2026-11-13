"""Fit free Groq requests without deleting evidence; recover wrapped quota errors."""
import copy
import json

GPT_OSS_MODEL = 'openai/gpt-oss-120b'
GPT_OSS_OUTPUT_TARGET = 4096


def output_limit(payload):
    values = [int(payload.get(key) or 0) for key in ('max_tokens', 'max_completion_tokens')]
    return max(values) or 2048


def desired_output_limit(payload):
    requested = output_limit(payload)
    if payload.get('model') == GPT_OSS_MODEL and requested >= 2048:
        return GPT_OSS_OUTPUT_TARGET
    return min(requested, 2048)


def adjusted_estimate(payload, estimate):
    # The earlier estimator always reserves at least 2048 output tokens.
    requested = output_limit(payload)
    return estimate(payload) - max(requested, 2048) + requested


def shorten_tool_help(value):
    changed = False
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'description' and isinstance(item, str) and len(item) > 180:
                value[key] = item[:177] + '...'
                changed = True
            elif isinstance(item, (dict, list)):
                changed = shorten_tool_help(item) or changed
    elif isinstance(value, list):
        for item in value:
            changed = shorten_tool_help(item) or changed
    return changed


def fit_payload(payload, limit, estimate, stop):
    result = copy.deepcopy(payload)
    original_requested = output_limit(result)
    requested = desired_output_limit(result)
    shortened = False
    input_and_overhead = adjusted_estimate(result, estimate) - output_limit(result)
    if input_and_overhead + requested + 64 > limit:
        shortened = shorten_tool_help(result.get('tools', []))
        input_and_overhead = adjusted_estimate(result, estimate) - output_limit(result)
    available = limit - input_and_overhead - 64
    stats = {'estimated_input_plus_overhead': input_and_overhead,
             'configured_token_budget': limit, 'message_count': len(result.get('messages', [])),
             'tool_count': len(result.get('tools', [])), 'tool_help_shortened': shortened,
             'original_output_tokens': original_requested, 'requested_output_tokens': requested}
    if available < min(512, requested):
        raise stop('GROQ_REQUEST_TOO_LARGE', {
            'quota_source': 'local_preflight', **stats,
            'estimated_request_tokens': adjusted_estimate(result, estimate),
            'quota_help': 'Input is still too large even after shortening tool help. Messages and evidence were not removed.'})
    allowance = min(requested, int(available))
    keys = [key for key in ('max_tokens', 'max_completion_tokens') if key in result] or ['max_tokens']
    for key in keys:
        result[key] = allowance
    stats.update(allowed_output_tokens=allowance, estimated_request_tokens=adjusted_estimate(result, estimate))
    if stats['estimated_request_tokens'] > limit:
        raise stop('GROQ_REQUEST_TOO_LARGE', {'quota_source': 'local_preflight', **stats})
    return result, stats


def recover_error(exc, info, stop_type):
    info = dict(info)
    seen = set()
    for _ in range(10):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        if isinstance(exc, stop_type):
            info.update(exc.details)
            info['underlying_error_type'] = 'QuotaStop'
            if str(exc.details.get('quota_source', '')).startswith('local_'):
                info.update(http_status=None, local_status=exc.code, status_is_local=True)
            break
        exc = exc.__cause__ or exc.__context__
    return info


def rebuild_request(request, payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    headers = {key: value for key, value in request.headers.items() if key.lower() != 'content-length'}
    return type(request)(request.method, request.url, headers=headers, content=body,
                         extensions=dict(request.extensions))


def install(free, quota):
    quota.install(free)
    original_send = quota.send_with_limits
    def send(send_function, client, request, args, kwargs, budget, **options):
        payload = json.loads(request.content)
        fitted, stats = fit_payload(payload, budget.limit, quota.estimate_tokens, quota.QuotaStop)
        print('GROQ_REQUEST_SIZE: ' + json.dumps(stats, sort_keys=True), flush=True)
        if fitted != payload:
            request = rebuild_request(request, fitted)
        options['estimate'] = lambda value: adjusted_estimate(value, quota.estimate_tokens)
        return original_send(send_function, client, request, args, kwargs, budget, **options)
    quota.send_with_limits = send
    original_details = free.error_details
    free.error_details = lambda exc: recover_error(exc, original_details(exc), quota.QuotaStop)


def self_test():
    import unittest
    class Stop(RuntimeError):
        def __init__(self, code, details):
            self.details = details
            super().__init__(code)
    def measured(input_tokens):
        return lambda payload: input_tokens + max(output_limit(payload), 2048)
    class Tests(unittest.TestCase):
        def payload(self, **changes):
            value = {'model': GPT_OSS_MODEL, 'max_tokens': 2048,
                     'messages': [{'role': 'user', 'content': 'Unmodified macro source evidence.'}]}
            value.update(changes)
            return value
        def test_gpt_oss_gets_more_output_when_it_fits(self):
            result, stats = fit_payload(self.payload(), 7500, measured(1331), Stop)
            self.assertEqual(result['max_tokens'], 4096)
            self.assertLessEqual(stats['estimated_request_tokens'], 7500)
        def test_logged_macro_size_gets_budget_safe_headroom(self):
            result, stats = fit_payload(self.payload(), 7500, measured(3921), Stop)
            self.assertEqual(result['max_tokens'], 3515)
            self.assertEqual(stats['estimated_request_tokens'], 7436)
        def test_evidence_and_original_payload_unchanged(self):
            payload = self.payload()
            saved = copy.deepcopy(payload)
            result, stats = fit_payload(payload, 7500, measured(3921), Stop)
            self.assertEqual(payload, saved)
            self.assertEqual(result['messages'], saved['messages'])
        def test_explicit_small_output_limit_is_preserved(self):
            result, stats = fit_payload(self.payload(max_tokens=512), 7500, measured(1331), Stop)
            self.assertEqual(result['max_tokens'], 512)
        def test_other_models_keep_original_ceiling(self):
            result, stats = fit_payload(self.payload(model='other-model'), 7500, measured(1331), Stop)
            self.assertEqual(result['max_tokens'], 2048)
        def test_completion_token_field_is_preserved(self):
            payload = self.payload()
            payload['max_completion_tokens'] = payload.pop('max_tokens')
            result, stats = fit_payload(payload, 7500, measured(3921), Stop)
            self.assertEqual(result['max_completion_tokens'], 3515)
            self.assertNotIn('max_tokens', result)
        def test_oversize_input_still_stops(self):
            with self.assertRaises(Stop):
                fit_payload(self.payload(), 7500, measured(7100), Stop)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


def main():
    from scripts import free_providers, free_groq_limits
    install(free_providers, free_groq_limits)
    return free_providers.main()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(0 if self_test() else 1)
    raise SystemExit(main())
