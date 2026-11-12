"""Offline unit tests only; no broker or model calls."""
import os
import unittest
from unittest.mock import patch
from scripts.free_actions import validate, redact, MODELS


class FreeActionsPolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(ALPACA_API_KEY='paper-key', ALPACA_SECRET_KEY='paper-secret',
                        ALPACA_EXPECTED_PAPER_ACCOUNT_ID='expected-paper-id',
                        ALPACA_USE_PAPER='True', GOOGLE_API_KEY='google-key',
                        SEC_USER_AGENT='Operator operator@example.com')

    def test_check_needs_no_ai_key(self):
        self.env.pop('GOOGLE_API_KEY')
        validate('AAPL', 'check', MODELS[0], False, self.env)

    def test_analysis_accepts_allowlisted_model(self):
        for model in MODELS:
            validate('BRK.B', 'analyze', model, True, self.env)

    def test_rejects_unconfirmed_billing(self):
        with self.assertRaises(ValueError):
            validate('AAPL', 'analyze', MODELS[0], False, self.env)

    def test_rejects_live_mode(self):
        self.env['ALPACA_USE_PAPER'] = 'False'
        with self.assertRaises(ValueError):
            validate('AAPL', 'check', MODELS[0], False, self.env)

    def test_rejects_paid_model(self):
        with self.assertRaises(ValueError):
            validate('AAPL', 'analyze', 'gemini-4-carbon', True, self.env)

    def test_rejects_multiple_symbols_and_shell_text(self):
        for symbol in ('AAPL,MSFT', 'aapl', 'AAPL;echo', '../AAPL', 'BTC/USD', ''):
            with self.assertRaises(ValueError):
                validate(symbol, 'check', MODELS[0], False, self.env)

    def test_rejects_missing_account_id(self):
        self.env.pop('ALPACA_EXPECTED_PAPER_ACCOUNT_ID')
        with self.assertRaises(ValueError):
            validate('AAPL', 'check', MODELS[0], False, self.env)

    def test_rejects_openai_key(self):
        self.env['OPENAI_API_KEY'] = 'not-allowed'
        with self.assertRaises(ValueError):
            validate('AAPL', 'check', MODELS[0], False, self.env)

    def test_rejects_order_mode(self):
        with self.assertRaises(ValueError):
            validate('AAPL', 'trade', MODELS[0], True, self.env)

    def test_redacts_keys_and_account_id(self):
        with patch.dict(os.environ, self.env, clear=True):
            text = redact('paper-key paper-secret expected-paper-id google-key')
        self.assertNotIn('paper-key', text)
        self.assertNotIn('expected-paper-id', text)
        self.assertEqual(text.count('[REDACTED]'), 4)


class NetworkPolicyTests(unittest.TestCase):
    def test_broker_writes_live_and_paid_hosts_are_blocked(self):
        import sys
        from types import SimpleNamespace
        from scripts.free_actions import install_http_guard
        class Session:
            def request(self, method, url, *args, **kwargs):
                return 'read-ok'
        class Client:
            def send(self, request, *args, **kwargs):
                return 'read-ok'
        class AsyncClient:
            async def send(self, request, *args, **kwargs):
                return 'read-ok'
        modules = {'requests': SimpleNamespace(sessions=SimpleNamespace(Session=Session)),
                   'httpx': SimpleNamespace(Client=Client, AsyncClient=AsyncClient)}
        with patch.dict(sys.modules, modules):
            check = install_http_guard()
            for method, url in [('POST', 'https://paper-api.alpaca.markets/v2/orders'),
                                ('DELETE', 'https://paper-api.alpaca.markets/v2/orders/1'),
                                ('GET', 'https://api.alpaca.markets/v2/account'),
                                ('POST', 'https://api.openai.com/v1/responses')]:
                with self.assertRaises(RuntimeError):
                    check(method, url)
            self.assertEqual(Session().request('GET', 'https://paper-api.alpaca.markets/v2/account'), 'read-ok')
            with self.assertRaises(RuntimeError):
                Session().request('POST', 'https://paper-api.alpaca.markets/v2/orders')

    def test_paper_identity_and_restrictions(self):
        import sys
        from types import SimpleNamespace
        from scripts.free_actions import paper_check
        env = dict(ALPACA_API_KEY='paper-key', ALPACA_SECRET_KEY='paper-secret',
                   ALPACA_EXPECTED_PAPER_ACCOUNT_ID='expected-paper-id')
        for payload, accepted in [({'id': 'expected-paper-id', 'status': 'ACTIVE'}, True),
                                  ({'id': 'wrong-id', 'status': 'ACTIVE'}, False),
                                  ({'id': 'expected-paper-id', 'status': 'ACTIVE', 'trading_blocked': True}, False)]:
            response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
            with patch.dict(sys.modules, {'requests': SimpleNamespace(get=lambda *a, **k: response)}), patch.dict(os.environ, env, clear=True):
                if accepted:
                    self.assertEqual(paper_check(), {'paper_account_verified': True, 'orders_enabled': False})
                else:
                    with self.assertRaises(ValueError):
                        paper_check()


if __name__ == '__main__':
    unittest.main()
