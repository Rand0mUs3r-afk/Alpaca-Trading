"""Groq schema compatibility and content-free provider error diagnostics.
Diagnostics do not retry requests, change trading checks or export raw errors.
Offline diagnostic checks: python -m scripts.free_schema_compat --self-test
"""
import copy
import json
import re

SCALAR_KEYS = {'type', 'enum', 'format', 'pattern', 'minLength', 'maxLength',
               'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
               'multipleOf', 'title', 'description', 'default', 'examples'}
ANNOTATIONS = {'title', 'description', 'default', 'examples'}
ERROR_CODES = {'tool_use_failed', 'invalid_request_error', 'json_validate_failed',
               'context_length_exceeded', 'model_not_found', 'invalid_api_key',
               'authentication_error', 'permission_error', 'invalid_tool_call'}
ERROR_TYPES = {'invalid_request_error', 'authentication_error', 'permission_error',
               'invalid_api_key', 'tool_use_failed', 'not_found_error'}
DIAGNOSTIC_KEY = 'tradingagents.safe_groq_error'


def resolve_local(branch, root):
    result = branch
    seen = set()
    while isinstance(result, dict) and '$ref' in result:
        ref = result['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/') or ref in seen:
            return None
        if set(result) - ({'$ref'} | ANNOTATIONS):
            return None
        seen.add(ref)
        result = root
        for part in ref[2:].split('/'):
            if not isinstance(result, dict):
                return None
            result = result.get(part.replace('~1', '/').replace('~0', '~'))
    return result


def normalize_nullable_scalars(schema):
    root = copy.deepcopy(schema)
    changes = 0
    def visit(node):
        nonlocal changes
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        branches = node.get('anyOf')
        if isinstance(branches, list) and len(branches) == 2 and set(node) <= ({'anyOf'} | ANNOTATIONS):
            resolved = [resolve_local(branch, root) for branch in branches]
            nulls = [item for item in resolved if isinstance(item, dict) and item.get('type') == 'null' and set(item) <= ({'type'} | ANNOTATIONS)]
            values = [item for item in resolved if isinstance(item, dict) and item.get('type') in ('string', 'number', 'integer', 'boolean') and set(item) <= SCALAR_KEYS]
            if len(nulls) == 1 and len(values) == 1:
                value = copy.deepcopy(values[0])
                enum = value.get('enum')
                if enum is None or (isinstance(enum, list) and all(isinstance(item, (str, int, float, bool)) or item is None for item in enum)):
                    annotations = {key: item for key, item in node.items() if key != 'anyOf'}
                    value['type'] = [value['type'], 'null']
                    if enum is not None:
                        value['enum'] = enum if None in enum else enum + [None]
                    value.update(annotations)
                    node.clear()
                    node.update(value)
                    changes += 1
        for child in list(node.values()):
            visit(child)
    visit(root)
    return root, changes


def normalize_payload(payload):
    result = copy.deepcopy(payload)
    response = result.get('response_format', {})
    if response.get('type') != 'json_schema':
        return result, 0
    spec = response.get('json_schema', {})
    if not isinstance(spec.get('schema'), dict):
        return result, 0
    spec['schema'], count = normalize_nullable_scalars(spec['schema'])
    return result, count


def compact_manager_history(text):
    if any('{' + name + '}' in text for name in ('risk_debate_digest', 'investment_debate_digest', 'research_debate_digest')):
        for name in ('history', 'bull_history', 'bear_history', 'risky_history', 'safe_history', 'neutral_history'):
            text = text.replace('{' + name + '}', '[Use the recent manager debate digest; full history is not repeated.]')
    return text


def request_shape(payload):
    messages = payload.get('messages') or []
    tools = payload.get('tools') or []
    pending, seen, issues, roles = set(), set(), set(), []
    call_count = result_count = 0
    for message in messages:
        if not isinstance(message, dict):
            issues.add('INVALID_MESSAGE_OBJECT')
            continue
        role = message.get('role')
        roles.append(role if role in ('system', 'developer', 'user', 'assistant', 'tool', 'function') else 'unknown')
        if role != 'tool' and pending:
            issues.add('TOOL_RESULTS_INTERRUPTED')
        if role == 'assistant':
            for call in message.get('tool_calls') or []:
                call_count += 1
                call_id = call.get('id') if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    issues.add('MISSING_TOOL_CALL_ID')
                else:
                    if call_id in seen:
                        issues.add('REUSED_TOOL_CALL_ID')
                    seen.add(call_id)
                    pending.add(call_id)
        elif role == 'tool':
            result_count += 1
            call_id = message.get('tool_call_id')
            if not isinstance(call_id, str) or call_id not in pending:
                issues.add('UNMATCHED_TOOL_RESULT')
            else:
                pending.remove(call_id)
    if pending:
        issues.add('MISSING_TOOL_RESULTS')
    names = {tool.get('function', {}).get('name') for tool in tools if isinstance(tool, dict) and isinstance(tool.get('function'), dict) and isinstance(tool['function'].get('name'), str)}
    analyst = 'unknown_or_no_tools'
    for label, markers in (('social', {'get_reddit_news', 'get_reddit_stock_info'}),
                           ('market', {'get_technical_brief', 'get_stockstats_indicators_report_online'}),
                           ('fundamentals', {'get_fundamentals_openai', 'get_simfin_balance_sheet'}),
                           ('macro', {'get_macro_analysis', 'get_macro_news_openai'}),
                           ('news', {'get_stock_news_openai', 'get_google_news', 'get_finnhub_news'})):
        if names & markers:
            analyst = label
            break
    choice = payload.get('tool_choice', 'auto')
    choice = 'named_function' if isinstance(choice, dict) else choice
    return {'analyst_from_tools': analyst, 'message_count': len(messages), 'roles_last_32': roles[-32:],
            'tool_count': len(tools), 'tool_call_count': call_count, 'tool_result_count': result_count,
            'tool_history_issues': sorted(issues),
            'tool_choice': choice if choice in ('auto', 'none', 'required', 'named_function') else 'other'}


def safe_diagnostic(response, payload):
    try:
        body = response.json()
        error = body.get('error', {}) if isinstance(body, dict) else {}
        error = error if isinstance(error, dict) else {}
    except Exception:
        error = {}
    code, kind = error.get('code'), error.get('type')
    code = code if isinstance(code, str) and code in ERROR_CODES else 'other_or_missing'
    kind = kind if isinstance(kind, str) and kind in ERROR_TYPES else 'other_or_missing'
    message = error.get('message', '')
    message = message[:8192].lower() if isinstance(message, str) else ''
    patterns = {
        'TOOL_GENERATION_FAILED': r'failed to call a function|failed to generate.*tool|tool_use_failed',
        'TOOL_HISTORY_REJECTED': r'tool_call_id|tool_calls.*followed|must be a response to',
        'TOOL_NOT_AVAILABLE': r'not.*provided tools|not in.*tools|tool.*not found',
        'SCHEMA_OR_OUTPUT_VALIDATION': r'schema|response_format|json_validate_failed',
        'UNSUPPORTED_ARGUMENT': r'unsupported|not supported|unknown parameter|unrecognized',
        'CONTEXT_LIMIT': r'context length|maximum context|too many tokens',
        'TOOL_CHOICE_REQUIREMENT': r'tool_choice|tool choice|required tool',
    }
    hints = [name for name, pattern in patterns.items() if re.search(pattern, message)]
    if code == 'tool_use_failed' and 'TOOL_GENERATION_FAILED' not in hints:
        hints.append('TOOL_GENERATION_FAILED')
    param = error.get('param')
    allowed = r'(messages|tools|response_format|tool_choice|max_tokens|max_completion_tokens|reasoning_effort|parallel_tool_calls|temperature|top_p|model)(?:\[\d+\]|\.(?:role|content|tool_calls|tool_call_id|function|name|arguments|type|json_schema|schema|strict))*'
    result = {'provider_error_code': code, 'provider_error_type': kind,
              'provider_error_hints': sorted(hints), 'request_shape': request_shape(payload),
              'diagnostic_scope': 'Allowlisted codes and structural metadata only; raw message and failed_generation omitted.'}
    if isinstance(param, str) and len(param) <= 128 and re.fullmatch(allowed, param):
        result['provider_error_param'] = param
    return result


def attach_diagnostic(response, payload):
    if getattr(response, 'status_code', None) in (400, 401, 403, 404, 422):
        try:
            extensions = getattr(response, 'extensions', None)
            if isinstance(extensions, dict):
                extensions[DIAGNOSTIC_KEY] = safe_diagnostic(response, payload)
        except Exception:
            # Observability must never replace the original provider exception.
            pass
    return response


def exception_diagnostic(exc):
    seen = set()
    for _ in range(12):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        response = getattr(exc, 'response', None)
        extensions = getattr(response, 'extensions', {})
        if isinstance(extensions, dict) and DIAGNOSTIC_KEY in extensions:
            return copy.deepcopy(extensions[DIAGNOSTIC_KEY])
        exc = exc.__cause__ or exc.__context__
    return {}


def install(free, compact, fit, quota):
    original_template = compact.compact_template
    compact.compact_template = lambda text: compact_manager_history(original_template(text))
    compact.install(free, fit, quota)
    original_send = quota.send_with_limits
    def send(send_function, client, request, args, kwargs, budget, **options):
        payload, count = normalize_payload(json.loads(request.content))
        if count:
            print('GROQ_NULLABLE_SCHEMA_FIELDS: ' + str(count))
            request = fit.rebuild_request(request, payload)
        response = original_send(send_function, client, request, args, kwargs, budget, **options)
        return attach_diagnostic(response, payload)
    quota.send_with_limits = send
    original_details = free.error_details
    def details(exc):
        info = original_details(exc)
        info.update(exception_diagnostic(exc))
        return info
    free.error_details = details


def self_test():
    import unittest
    from types import SimpleNamespace as NS
    class Tests(unittest.TestCase):
        def response(self, code=400, error=None):
            return NS(status_code=code, extensions={}, json=lambda: {'error': error or {}})
        def test_private_text_and_generation_never_exported(self):
            response = self.response(error={'code': 'tool_use_failed', 'type': 'invalid_request_error',
                'message': 'Failed to call a function secret-example', 'failed_generation': 'private-account-example'})
            result = safe_diagnostic(response, {'messages': [{'role': 'user', 'content': 'private-prompt-example'}]})
            text = json.dumps(result)
            for value in ('secret-example', 'private-account-example', 'private-prompt-example'):
                self.assertNotIn(value, text)
            self.assertIn('TOOL_GENERATION_FAILED', result['provider_error_hints'])
        def test_correct_tool_pairing(self):
            shape = request_shape({'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'a'}]},
                {'role': 'tool', 'tool_call_id': 'a'}]})
            self.assertEqual(shape['tool_history_issues'], [])
        def test_missing_and_unmatched_results(self):
            shape = request_shape({'messages': [{'role': 'assistant', 'tool_calls': [{'id': 'a'}]},
                {'role': 'tool', 'tool_call_id': 'b'}]})
            self.assertEqual(shape['tool_history_issues'], ['MISSING_TOOL_RESULTS', 'UNMATCHED_TOOL_RESULT'])
        def test_unknown_error_fields_not_exported(self):
            result = safe_diagnostic(self.response(error={'code': 'private-code', 'type': 'private-type', 'param': 'private-param'}), {})
            self.assertNotIn('private-', json.dumps(result))
        def test_safe_parameter_path_is_retained(self):
            result = safe_diagnostic(self.response(error={'param': 'messages[2].tool_call_id'}), {})
            self.assertEqual(result['provider_error_param'], 'messages[2].tool_call_id')
        def test_failed_response_keeps_diagnostic_on_its_own_exception(self):
            response = self.response(error={'code': 'tool_use_failed'})
            self.assertIs(attach_diagnostic(response, {}), response)
            cause = ValueError(); cause.response = response
            error = RuntimeError(); error.__cause__ = cause
            self.assertEqual(exception_diagnostic(error)['provider_error_code'], 'tool_use_failed')
            self.assertEqual(exception_diagnostic(RuntimeError()), {})
        def test_successful_response_unchanged(self):
            response = self.response(code=200)
            attach_diagnostic(response, {})
            self.assertEqual(response.extensions, {})
        def test_schema_constraints_retained(self):
            schema = {'type': 'object', 'properties': {'action': {'type': 'string', 'enum': ['BUY', 'HOLD']},
                'stop': {'anyOf': [{'type': 'number'}, {'type': 'null'}]}}, 'additionalProperties': False}
            original = copy.deepcopy(schema)
            result, count = normalize_nullable_scalars(schema)
            self.assertEqual(count, 1)
            self.assertEqual(schema, original)
            self.assertEqual(result['properties']['action'], schema['properties']['action'])
            self.assertFalse(result['additionalProperties'])
        def test_install_exports_attached_details_without_changing_response(self):
            response = self.response(error={'code': 'tool_use_failed'})
            free = NS(error_details=lambda exc: {'error_code': 'AI_INVALID_REQUEST'})
            compact = NS(compact_template=lambda value: value, install=lambda *args: None)
            quota = NS(send_with_limits=lambda *args, **kwargs: response)
            install(free, compact, NS(), quota)
            request = NS(content=json.dumps({'messages': []}).encode())
            self.assertIs(quota.send_with_limits(None, None, request, (), {}, None), response)
            error = ValueError(); error.response = response
            result = free.error_details(error)
            self.assertEqual(result['error_code'], 'AI_INVALID_REQUEST')
            self.assertEqual(result['provider_error_code'], 'tool_use_failed')
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


def main():
    from scripts import free_providers, free_compact_context, free_request_fit, free_groq_limits
    install(free_providers, free_compact_context, free_request_fit, free_groq_limits)
    return free_providers.main()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(0 if self_test() else 1)
    raise SystemExit(main())
