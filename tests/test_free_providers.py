"""Offline provider routing tests; no real model or brokerage calls."""
import json
import unittest
from types import SimpleNamespace
from scripts import free_providers as free


class APIError(Exception):
    def __init__(self, code):
        super().__init__('private-value-do-not-print')
        self.code = code


class ProviderTests(unittest.TestCase):
    def setUp(self):
        free.ATTEMPTS.clear()
        free.ACTIVE.clear()

    def test_routes_gemini_then_groq(self):
        self.assertEqual(free.routes('auto', free.GEMINI_MODELS[0], {'GOOGLE_API_KEY': 'g', 'GROQ_API_KEY': 'q'}),
                         [('google', free.GEMINI_MODELS[0]), ('groq', free.GROQ_MODEL)])

    def test_groq_needs_no_google_key(self):
        self.assertEqual(free.routes('groq', free.GEMINI_MODELS[0], {'GROQ_API_KEY': 'q'}), [('groq', free.GROQ_MODEL)])

    def test_missing_key_is_rejected(self):
        with self.assertRaises(ValueError):
            free.routes('groq', free.GEMINI_MODELS[0], {})

    def test_fallback_restarts_after_server_error(self):
        calls = []
        def run(provider, model):
            calls.append(provider)
            if provider == 'google':
                raise APIError(503)
            return 'finished'
        result = free.choose_analysis([('google', free.GEMINI_MODELS[0]), ('groq', free.GROQ_MODEL)], run)
        self.assertEqual(result, 'finished')
        self.assertEqual(calls, ['google', 'groq'])
        self.assertEqual(free.ACTIVE['provider'], 'groq')
        self.assertEqual(free.ATTEMPTS[0]['http_status'], 503)

    def test_no_fallback_for_auth_or_code_errors(self):
        for error in (APIError(401), APIError(403), ValueError('bug')):
            calls = []
            def run(provider, model):
                calls.append(provider)
                raise error
            with self.assertRaises(type(error)):
                free.choose_analysis([('google', free.GEMINI_MODELS[0]), ('groq', free.GROQ_MODEL)], run)
            self.assertEqual(calls, ['google'])

    def test_quota_can_switch_provider(self):
        self.assertTrue(free.can_fallback(APIError(429)))
        self.assertTrue(free.can_fallback(APIError(404)))

    def test_retry_is_bounded(self):
        calls, waits = [], []
        def fail():
            calls.append(1)
            raise APIError(503)
        with self.assertRaises(APIError):
            free.bounded_call(fail, sleep=waits.append)
        self.assertEqual(len(calls), 2)
        self.assertEqual(waits, [20])

    def test_does_not_repeat_quota_or_invalid_requests(self):
        for code in (429, 400, 401):
            waits = []
            def fail():
                raise APIError(code)
            with self.assertRaises(APIError):
                free.bounded_call(fail, sleep=waits.append)
            self.assertEqual(waits, [])

    def test_only_approved_groq_endpoint(self):
        free.check_url('POST', free.GROQ_URL + '/chat/completions')
        for method, url in [('POST', 'https://api.openai.com/v1/chat/completions'),
                            ('GET', 'https://api.alpaca.markets/v2/account'),
                            ('POST', 'https://paper-api.alpaca.markets/v2/orders'),
                            ('POST', free.GROQ_URL + '/responses')]:
            with self.assertRaises(RuntimeError):
                free.check_url(method, url)
        free.check_url('GET', 'https://paper-api.alpaca.markets/v2/account')

    def test_builds_groq_with_correct_secret_and_endpoint(self):
        def constructor(**kwargs):
            return kwargs
        result = free.build_llm('groq', free.GROQ_MODEL, constructor, constructor, {'GROQ_API_KEY': 'groq-key'})
        self.assertEqual(result['api_key'], 'groq-key')
        self.assertEqual(result['base_url'], free.GROQ_URL)
        self.assertFalse(result['use_responses_api'])
        self.assertEqual(result['max_retries'], 0)

    def test_disallows_other_models(self):
        with self.assertRaises(ValueError):
            free.build_llm('groq', 'unapproved-model', dict, dict, {'GROQ_API_KEY': 'key'})

    def test_status_extraction_from_cause(self):
        outer = RuntimeError('hidden')
        outer.__cause__ = APIError(503)
        self.assertEqual(free.http_status(outer), 503)

    def test_error_report_never_includes_exception_text(self):
        result = json.dumps(free.error_details(APIError(503)))
        self.assertNotIn('private-value', result)
        self.assertEqual(json.loads(result)['error_code'], 'AI_SERVER_BUSY')


    def test_installs_groq_adapter_without_google_credentials(self):
        import sys
        from unittest.mock import patch
        class FakeChat:
            def __init__(self, **kwargs):
                self.options = kwargs
            def _generate(self, *args, **kwargs):
                raise AssertionError('No real generation in this test')
        graph_module = SimpleNamespace(TradingAgentsGraph=object)
        memory_class = type('LocalMemory', (), {})
        modules = {
            'langchain_openai': SimpleNamespace(ChatOpenAI=FakeChat),
            'tradingagents': SimpleNamespace(),
            'tradingagents.llm_clients': SimpleNamespace(),
            'tradingagents.llm_clients.google_client': SimpleNamespace(NormalizedChatGoogleGenerativeAI=FakeChat),
            'tradingagents.graph': SimpleNamespace(trading_graph=graph_module),
            'tradingagents.agents': SimpleNamespace(),
            'tradingagents.agents.utils': SimpleNamespace(memory=SimpleNamespace(FinancialSituationMemory=memory_class)),
        }
        with patch.dict(sys.modules, modules), patch.dict(__import__('os').environ, {'GROQ_API_KEY': 'groq-test-key'}, clear=True):
            self.assertIs(free.install_clients(), object)
            llm = graph_module.create_llm_client('groq', free.GROQ_MODEL, model_role='quick').get_llm()
            self.assertEqual(llm.options['api_key'], 'groq-test-key')
            self.assertEqual(llm.options['base_url'], free.GROQ_URL)
            self.assertIs(graph_module.FinancialSituationMemory, memory_class)

    def test_main_groq_only_with_mocked_graph(self):
        import importlib
        import os
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        # Reproduce full-suite order even when this test runs alone.
        importlib.import_module('scripts.free_actions')
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'result.json'
            base = SimpleNamespace(
                REPORT=report, REPORTS=('market_report',),
                validate=Mock(return_value=None),
                paper_check=lambda: {'paper_account_verified': True, 'orders_enabled': False},
                install_free_tools=lambda: None, install_local_memory=lambda: None,
                redact=lambda text: text)
            class FakeGraph:
                def propagate(self, ticker, date):
                    return {'market_report': 'mock research'}, 'HOLD'
            modules = {
                'scripts.free_actions': base,
                'tradingagents.dataflows.config': SimpleNamespace(clear_runtime_api_keys=lambda: None),
            }
            env = {'GROQ_API_KEY': 'groq-test-key', 'SEC_USER_AGENT': 'Test test@example.com'}
            argv = ['runner', '--mode', 'analyze', '--provider', 'groq', '--free-tier-confirmed', 'true']
            with patch.dict(sys.modules, modules), patch('scripts.free_actions', base, create=True), patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), patch.object(free, 'install_http_guard', lambda: None), patch.object(free, 'install_clients', lambda: FakeGraph), patch.object(free, 'run_graph', lambda *args: FakeGraph()):
                self.assertEqual(free.main(), 0)
            base.validate.assert_called_once_with('AAPL', 'check', free.GEMINI_MODELS[0], False)
            result = json.loads(report.read_text())
            self.assertEqual(result['provider'], 'groq')
            self.assertEqual(result['model'], free.GROQ_MODEL)
            self.assertEqual(result['status'], 'ANALYSIS_COMPLETE_NO_ORDERS')
            self.assertFalse(result['orders_enabled'])
            self.assertNotIn('groq-test-key', report.read_text())


if __name__ == '__main__':
    unittest.main()
