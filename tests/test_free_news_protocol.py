"""Offline news regressions using the existing social-protocol stand-ins.
Exercises production function bodies, not real SDK serialization or providers.
Run with the repository's unittest discovery command.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from test_free_social_protocol import Message, Prompt, Tool, Model


def call(call_id='call_1', name='get_google_news'):
    return {'id': call_id, 'name': name, 'args': {'query': 'AAPL stock', 'curr_date': '2026-11-12'}}


class NewsProtocolTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'tradingagents/agents/analysts/news_analyst.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        names = {'_news_tool_calls', '_news_text', 'create_news_analyst'}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        self.bound_names = []
        self.ns = {'json': json, 'AIMessage': Message, 'ToolMessage': Message,
            'ChatPromptTemplate': Prompt, 'MessagesPlaceholder': lambda **kwargs: None,
            'load_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'render_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'capture_agent_prompt': lambda *args, **kwargs: None}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), self.ns)

    def fixture(self, replies, ticker='AAPL', finnhub=True, global_news=False, openai=True, iterations=8):
        tools = {name: Tool(name) for name in ('get_google_news', 'get_finnhub_news_recent',
                  'get_coindesk_news', 'get_global_news_openai')}
        toolkit = SimpleNamespace(config={'online_tools': True, 'news_global_openai_enabled': global_news,
            'max_tool_iterations_per_agent': iterations}, has_openai_web_search=lambda: openai,
            has_finnhub=lambda: finnhub, has_coindesk=lambda: True, **tools)
        model = Model(replies + [Message(content='Fixture report. FINAL TRANSACTION PROPOSAL: HOLD')])
        original_bind = model.bind_tools
        def bind(selected):
            self.bound_names = [tool.name for tool in selected]
            return original_bind(selected)
        model.bind_tools = bind
        node = self.ns['create_news_analyst'](model, toolkit)
        state = {'trade_date': '2026-11-12', 'company_of_interest': ticker, 'messages': []}
        return node, state, model, tools

    def test_canonical_and_legacy_calls(self):
        raw = {'id': 'call_1', 'type': 'function', 'function': {'name': call()['name'], 'arguments': json.dumps(call()['args'])}}
        for first in (Message(tool_calls=[call()]), Message(additional_kwargs={'tool_calls': [raw]})):
            with self.subTest(format='canonical' if first.tool_calls else 'legacy'):
                node, state, model, tools = self.fixture([first])
                node(state)
                self.assertIs(model.histories[1][0], first)
                self.assertEqual(model.histories[1][1].tool_call_id, 'call_1')
                self.assertEqual(len(tools['get_google_news'].calls), 1)
                self.assertEqual(state['messages'], [])

    def test_multi_call_message_is_preserved_once(self):
        first = Message(tool_calls=[call(), call('call_2', 'get_finnhub_news_recent')])
        node, state, model, tools = self.fixture([first])
        node(state)
        self.assertEqual(len(model.histories[1]), 3)
        self.assertIs(model.histories[1][0], first)
        self.assertEqual(first.response_metadata, {'fixture': 'preserve-me'})
        self.assertEqual([item.tool_call_id for item in model.histories[1][1:]], ['call_1', 'call_2'])

    def test_invalid_batches_execute_no_tools(self):
        cases = [([{**call(), 'id': None}], 'NAME_OR_ID'), ([call(), call()], 'DUPLICATE'),
            ([{**call(), 'args': 'not-json'}], 'ARGUMENTS'), ([{**call(), 'args': []}], 'ARGUMENTS'),
            ([{**call(), 'args': {'x': float('nan')}}], 'ARGUMENTS'),
            ([call(), call('call_2', 'unknown_tool')], 'UNKNOWN')]
        for calls, code in cases:
            with self.subTest(code=code):
                node, state, model, tools = self.fixture([Message(tool_calls=calls)])
                with self.assertRaisesRegex(ValueError, code):
                    node(state)
                self.assertTrue(all(not tool.calls for tool in tools.values()))

    def test_caching_and_reused_ids(self):
        node, state, model, tools = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])])
        node(state)
        self.assertEqual(len(tools['get_google_news'].calls), 1)
        node, state, model, tools = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call()])])
        with self.assertRaisesRegex(ValueError, 'NEWS_REUSED_TOOL_CALL_ID'):
            node(state)
        self.assertEqual(len(tools['get_google_news'].calls), 1)

    def test_empty_narrative_remains_explicit(self):
        node, state, model, tools = self.fixture([Message()])
        report = node(state)['news_report']
        self.assertIn('did not return a complete news narrative', report)
        self.assertIn('FINAL TRANSACTION PROPOSAL:', report)
        self.assertEqual(self.ns['_news_text']([
            {'type': 'text', 'text': 'Visible'}, {'type': 'reasoning', 'text': 'Hidden'}]), 'Visible')

    def test_private_tool_exception_text_not_forwarded(self):
        node, state, model, tools = self.fixture([Message(tool_calls=[call()])])
        tools['get_google_news'].fail = True
        node(state)
        self.assertIn('RuntimeError', model.histories[1][1].content)
        self.assertNotIn('private-credential-fixture', model.histories[1][1].content)

    def test_source_selection_unchanged(self):
        cases = [({'finnhub': False}, ['get_google_news']),
            ({'ticker': 'BTC/USD'}, ['get_google_news', 'get_coindesk_news']),
            ({'global_news': True}, ['get_google_news', 'get_global_news_openai', 'get_finnhub_news_recent']),
            ({'global_news': True, 'openai': False}, ['get_google_news', 'get_finnhub_news_recent'])]
        for options, expected in cases:
            with self.subTest(options=options):
                node, state, model, tools = self.fixture([Message(content='FINAL TRANSACTION PROPOSAL: HOLD')], **options)
                node(state)
                self.assertEqual(self.bound_names, expected)

    def test_tool_loop_limit_remains_explicit(self):
        node, state, model, tools = self.fixture(
            [Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])], iterations=1)
        report = node(state)['news_report']
        self.assertIn('Tool-loop halted after 1 iterations', report)
        self.assertEqual(len(tools['get_google_news'].calls), 1)
        self.assertEqual(len(model.histories), 3)


if __name__ == '__main__':
    unittest.main()
