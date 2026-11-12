"""Offline prompt-profile tests; no provider or brokerage requests."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts import free_compact_context as compact


BULL = ('Bull role.\nMatrix: {claim_matrix}\nFull reports: {all_reports_text}\n'
        'Digest: {debate_digest}\nFull history: {history}\nLast bear: {current_response}\n')


class CompactTests(unittest.TestCase):
    def test_drops_duplicate_full_blocks_but_keeps_brief_and_counterargument(self):
        result = compact.compact_template(BULL)
        self.assertNotIn('{all_reports_text}', result)
        self.assertNotIn('{history}', result)
        for field in ('claim_matrix', 'debate_digest', 'current_response'):
            self.assertIn('{' + field + '}', result)
        rendered = result.format_map(dict(claim_matrix='Revenue=123; stop=95', debate_digest='Bear disagrees', current_response='Demand fell'))
        self.assertIn('Revenue=123; stop=95', rendered)
        self.assertIn('Demand fell', rendered)

    def test_does_not_remove_reports_without_evidence_replacement(self):
        original = 'Only source: {all_reports_text}'
        self.assertEqual(compact.compact_template(original), original)

    def test_latest_responses_and_decision_format_are_preserved(self):
        template = '{claim_matrix} {all_reports_text} {debate_digest} {history} {current_risky_response} {current_safe_response} {decision_format}'
        result = compact.compact_template(template)
        for field in ('current_risky_response', 'current_safe_response', 'decision_format'):
            self.assertIn('{' + field + '}', result)

    def test_large_reports_are_not_repeated_in_rendered_prompt(self):
        values = dict(claim_matrix='selected claims', all_reports_text='full report ' * 5000,
                      debate_digest='recent debate', history='history ' * 3000, current_response='latest objection')
        before = BULL.format_map(values)
        after = compact.compact_template(BULL).format_map(values)
        self.assertLess(len(after), len(before) // 10)

    def test_profile_copies_unchanged_templates_too(self):
        templates = {'researchers/bull_researcher.md': BULL, 'shared/other.md': 'unchanged'}
        loader = SimpleNamespace(list_prompt_templates=lambda: list(templates), load_prompt=lambda name: templates[name])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(compact.write_profile(loader, directory), 1)
            self.assertEqual((Path(directory) / 'shared/other.md').read_text(), 'unchanged')

    def test_scope_restores_environment_and_defaults_after_failure(self):
        templates = {'researchers/bull_researcher.md': BULL}
        loader = SimpleNamespace(list_prompt_templates=lambda: list(templates), load_prompt=lambda name: templates[name])
        context = SimpleNamespace(DEFAULT_CONTEXT_CONFIG={'report_context_budget_tokens': 5500, 'other': True})
        saved = dict(context.DEFAULT_CONTEXT_CONFIG)
        def fail(*args, **kwargs):
            self.assertEqual(context.DEFAULT_CONTEXT_CONFIG['report_context_budget_tokens'], 1600)
            self.assertTrue(Path(os.environ['TRADINGAGENTS_PROMPT_DIR']).exists())
            raise ValueError('simulated failure')
        modules = {'tradingagents.prompts.loader': loader,
                   'tradingagents.agents.utils.report_context': context}
        with patch.dict(os.environ, {'TRADINGAGENTS_PROMPT_DIR': '/previous'}), patch.object(compact.importlib, 'import_module', side_effect=lambda name: modules[name]):
            with self.assertRaises(ValueError):
                compact.ProfileGraph(SimpleNamespace(propagate=fail)).propagate('AAPL', '2026-11-12')
            self.assertEqual(os.environ['TRADINGAGENTS_PROMPT_DIR'], '/previous')
            self.assertEqual(context.DEFAULT_CONTEXT_CONFIG, saved)

    def test_profile_keeps_roles_and_round_settings(self):
        original_config = {'selected_analysts': ['market', 'social', 'news', 'fundamentals', 'macro'],
                           'max_debate_rounds': 3, 'max_risk_discuss_rounds': 3}
        free = SimpleNamespace(run_graph=lambda provider, model, cls: cls(config=original_config))
        fit = SimpleNamespace(install=lambda *args: None)
        seen = []
        compact.install(free, fit, object())
        free.run_graph('groq', 'model', lambda config: seen.append(config))
        self.assertEqual(seen[0]['selected_analysts'], original_config['selected_analysts'])
        self.assertEqual(seen[0]['max_debate_rounds'], 3)
        self.assertEqual(seen[0]['max_risk_discuss_rounds'], 3)
        self.assertEqual(seen[0]['report_context_budget_tokens'], 1600)
        self.assertNotIn('report_context_budget_tokens', original_config)

    def test_google_graph_is_not_wrapped(self):
        graph = object()
        free = SimpleNamespace(run_graph=lambda *args: graph)
        compact.install(free, SimpleNamespace(install=lambda *args: None), object())
        self.assertIs(free.run_graph('google', 'model', object), graph)


if __name__ == '__main__':
    unittest.main()
