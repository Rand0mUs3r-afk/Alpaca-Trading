"""Offline request-shaping and wrapped-error regressions."""
import json
import unittest
from scripts import free_request_fit as fit


class Stop(RuntimeError):
    def __init__(self, code, details):
        super().__init__(code)
        self.code = 413
        self.details = {'error_code': code, **details}


def estimate(payload):
    text = json.dumps(payload.get('messages', [])) + json.dumps(payload.get('tools', []))
    return len(text) + max(fit.output_limit(payload), 2048) + 256


class FitTests(unittest.TestCase):
    def test_recovers_local_stop_wrapped_by_sdk(self):
        original = Stop('GROQ_REQUEST_TOO_LARGE', {
            'quota_source': 'local_preflight', 'estimated_request_tokens': 9000,
            'configured_token_budget': 7500})
        wrapper = RuntimeError('Connection error. private-value')
        wrapper.__cause__ = original
        report = fit.recover_error(wrapper, {'http_status': 413}, Stop)
        self.assertEqual(report['error_code'], 'GROQ_REQUEST_TOO_LARGE')
        self.assertEqual(report['estimated_request_tokens'], 9000)
        self.assertTrue(report['status_is_local'])
        self.assertIsNone(report['http_status'])
        self.assertNotIn('private-value', json.dumps(report))

    def test_reduces_output_without_deleting_messages(self):
        payload = {'messages': [{'role': 'user', 'content': 'evidence' * 120}], 'max_tokens': 2048}
        result, stats = fit.fit_payload(payload, 2600, estimate, Stop)
        self.assertLess(result['max_tokens'], 2048)
        self.assertGreaterEqual(result['max_tokens'], 512)
        self.assertEqual(result['messages'], payload['messages'])
        self.assertLessEqual(stats['estimated_request_tokens'], 2600)
        self.assertEqual(payload['max_tokens'], 2048)

    def test_shortens_only_tool_help_not_schema_or_facts(self):
        tool = {'type': 'function', 'function': {'name': 'get_data', 'description': 'help ' * 700,
                'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}}, 'required': ['ticker']}}}
        payload = {'messages': [{'role': 'user', 'content': 'Revenue=123; price=100'}], 'tools': [tool], 'max_tokens': 2048}
        result, stats = fit.fit_payload(payload, 3000, estimate, Stop)
        self.assertTrue(stats['tool_help_shortened'])
        self.assertEqual(result['messages'], payload['messages'])
        self.assertEqual(result['tools'][0]['function']['parameters'], tool['function']['parameters'])
        self.assertEqual(result['tools'][0]['function']['name'], 'get_data')
        self.assertLessEqual(len(result['tools'][0]['function']['description']), 180)

    def test_stops_when_input_itself_cannot_fit(self):
        payload = {'messages': [{'role': 'user', 'content': 'x' * 9000}], 'max_tokens': 2048}
        with self.assertRaises(Stop) as caught:
            fit.fit_payload(payload, 7500, estimate, Stop)
        self.assertEqual(caught.exception.details['quota_source'], 'local_preflight')
        self.assertEqual(caught.exception.details['message_count'], 1)

    def test_handles_max_completion_tokens(self):
        payload = {'messages': [{'role': 'user', 'content': 'x' * 1000}], 'max_completion_tokens': 2048}
        result, stats = fit.fit_payload(payload, 2600, estimate, Stop)
        self.assertNotIn('max_tokens', result)
        self.assertLess(result['max_completion_tokens'], 2048)
        self.assertLessEqual(stats['estimated_request_tokens'], 2600)

    def test_does_not_relabel_remote_status_as_local(self):
        original = Stop('GROQ_REQUEST_TOO_LARGE', {'quota_source': 'groq_response'})
        report = fit.recover_error(original, {'http_status': 413}, Stop)
        self.assertEqual(report['http_status'], 413)
        self.assertNotIn('status_is_local', report)

    def test_cause_cycles_are_bounded(self):
        error = RuntimeError('private')
        error.__cause__ = error
        self.assertEqual(fit.recover_error(error, {'error_code': 'unknown'}, Stop), {'error_code': 'unknown'})

    def test_rebuild_preserves_headers_and_extensions(self):
        class Request:
            def __init__(self, method, url, headers, content, extensions):
                self.method, self.url, self.headers = method, url, headers
                self.content, self.extensions = content, extensions
        original = Request('POST', 'https://api.groq.com/openai/v1/chat/completions',
                           {'Authorization': 'Bearer private', 'Content-Length': '999', 'Content-Type': 'application/json'},
                           b'{}', {'timeout': {'read': 120}})
        result = fit.rebuild_request(original, {'max_tokens': 512})
        self.assertNotIn('Content-Length', result.headers)
        self.assertEqual(result.headers['Authorization'], original.headers['Authorization'])
        self.assertEqual(result.extensions, original.extensions)
        self.assertEqual(json.loads(result.content), {'max_tokens': 512})


if __name__ == '__main__':
    unittest.main()
