"""Offline tests for the diagnostic helper; no AI or broker requests."""
import json
import unittest
from scripts import free_actions_debug as debug


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        debug.STAGE = 'startup'
        debug.FAILURE = None

    def test_aliases_only_add_equivalent_resource_names(self):
        names = debug.model_aliases(('gemini-3.8-flash', 'gemini-3.6-flash'))
        self.assertEqual(set(names), {'gemini-3.8-flash', 'gemini-3.6-flash',
                                    'models/gemini-3.8-flash', 'models/gemini-3.6-flash'})
        self.assertNotIn('gemini-pro-paid', names)

    def test_aliases_are_idempotent(self):
        once = debug.model_aliases(('gemini-3.8-flash',))
        self.assertEqual(debug.model_aliases(once), once)

    def test_successful_wrapper_keeps_return_value(self):
        self.assertEqual(debug.observed('local_embeddings', lambda: 42)(), 42)
        self.assertEqual(debug.STAGE, 'local_embeddings')
        self.assertIsNone(debug.FAILURE)

    def test_unknown_exception_text_is_never_published(self):
        def fail():
            raise ValueError('private-token-abcdef private-account-balance=12345')
        with self.assertRaises(ValueError):
            debug.observed('graph_initialization', fail)()
        result = debug.enrich_result(json.dumps({'status': 'FAILED', 'orders_enabled': False}))
        self.assertNotIn('private-token', result)
        self.assertNotIn('12345', result)
        self.assertEqual(json.loads(result)['error_stage'], 'graph_initialization')
        self.assertTrue(json.loads(result)['error_location'])

    def test_known_error_has_specific_code(self):
        debug.capture_failure(ValueError("Model is not on this runner's allowlist."))
        self.assertEqual(debug.FAILURE['error_code'], 'MODEL_NAME_FORMAT')

    def test_preserves_paper_verification_and_order_block(self):
        result = json.loads(debug.enrich_result(json.dumps({
            'status': 'FAILED', 'paper_account_verified': True, 'orders_enabled': False})))
        self.assertTrue(result['paper_account_verified'])
        self.assertFalse(result['orders_enabled'])

    def test_non_json_is_unchanged(self):
        self.assertEqual(debug.enrich_result('not-json'), 'not-json')


if __name__ == '__main__':
    unittest.main()
