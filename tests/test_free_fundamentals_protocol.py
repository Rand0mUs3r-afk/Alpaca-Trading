"""Offline fundamentals regressions with stand-in SDK messages and tools.
Tests production function bodies, not real SDK serialization or providers.
"""
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from test_free_social_protocol import Message, Prompt, Tool, Model

NAMES = ('get_fundamentals_openai', 'get_finnhub_company_insider_sentiment',
         'get_finnhub_company_insider_transactions', 'get_simfin_balance_sheet',
         'get_simfin_cashflow', 'get_simfin_income_stmt', 'get_defillama_fundamentals')


class RaisingModel(Model):
    def invoke(self, history):
        if self.replies and isinstance(self.replies[0], Exception):
            self.histories.append(list(history) if isinstance(history, list) else history)
            raise self.replies.pop(0)
        return super().invoke(history)


def call(call_id='call_1', name='get_fundamentals_openai'):
    return {'id': call_id, 'name': name, 'args': {'ticker': 'AAPL', 'curr_date': '2026-11-12'}}


class FundamentalsProtocolTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'tradingagents/agents/analysts/fundamentals_analyst.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        names = {'_fundamentals_tool_calls', '_fundamentals_text', 'create_fundamentals_analyst'}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        self.ns = {'json': json, 'AIMessage': Message, 'ToolMessage': Message,
            'ChatPromptTemplate': Prompt, 'MessagesPlaceholder': lambda **kwargs: None,
            'load_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'render_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'capture_agent_prompt': lambda *args, **kwargs: None}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), self.ns)

    def fixture(self, replies, ticker='AAPL', openai=True, finnhub=False, simfin=False, iterations=8):
        tools = {name: Tool(name) for name in NAMES}
        toolkit = SimpleNamespace(config={'online_tools': True, 'max_tool_iterations_per_agent': iterations},
            has_openai_web_search=lambda: openai, has_finnhub=lambda: finnhub,
            has_simfin_data=lambda: simfin, **tools)
        model = RaisingModel(replies + [Message(content='Fixture report. FINAL TRANSACTION PROPOSAL: HOLD')])
        self.bound_names = []
        original_bind = model.bind_tools
        def bind(selected):
            self.bound_names = [tool.name for tool in selected]
            return original_bind(selected)
        model.bind_tools = bind
        node = self.ns['create_fundamentals_analyst'](model, toolkit)
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
                self.assertEqual(len(tools[NAMES[0]].calls), 1)
                self.assertEqual(state['messages'], [])

    def test_multiple_calls_preserve_one_original_reply(self):
        first = Message(tool_calls=[call(), call('call_2', 'get_simfin_balance_sheet')])
        node, state, model, tools = self.fixture([first], simfin=True)
        report = node(state)
        self.assertEqual(len(model.histories[1]), 3)
        self.assertIs(model.histories[1][0], first)
        self.assertEqual(first.response_metadata, {'fixture': 'preserve-me'})
        self.assertEqual([item.tool_call_id for item in model.histories[1][1:]], ['call_1', 'call_2'])
        self.assertEqual(len(report['messages']), 4)

    def test_bad_batches_execute_no_tools(self):
        cases = [([{**call(), 'id': None}], 'NAME_OR_ID'), ([call(), call()], 'DUPLICATE'),
            ([{**call(), 'args': 'not-json'}], 'ARGUMENTS'), ([{**call(), 'args': []}], 'ARGUMENTS'),
            ([{**call(), 'args': {'x': float('nan')}}], 'ARGUMENTS'),
            ([call(), call('call_2', 'unknown_tool')], 'UNKNOWN')]
        for calls, code in cases:
            with self.subTest(code=code), redirect_stdout(StringIO()):
                node, state, model, tools = self.fixture([Message(tool_calls=calls)])
                with self.assertRaisesRegex(ValueError, code):
                    node(state)
                self.assertTrue(all(not tool.calls for tool in tools.values()))

    def test_cached_calls_and_reused_ids(self):
        node, state, model, tools = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])])
        node(state)
        self.assertEqual(len(tools[NAMES[0]].calls), 1)
        node, state, model, tools = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call()])])
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'FUNDAMENTALS_REUSED_TOOL_CALL_ID'):
            node(state)
        self.assertEqual(len(tools[NAMES[0]].calls), 1)

    def test_provider_failures_propagate_before_and_after_data(self):
        for after_data in (False, True):
            with self.subTest(after_data=after_data):
                error = RuntimeError('private-provider-credential-fixture')
                error.response = SimpleNamespace(extensions={'safe_fixture': 'preserved'})
                replies = [Message(tool_calls=[call()]), error] if after_data else [error]
                node, state, model, tools = self.fixture(replies)
                output = StringIO()
                with redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
                    node(state)
                self.assertIs(caught.exception, error)
                self.assertIs(caught.exception.response, error.response)
                self.assertNotIn('private-provider-credential-fixture', output.getvalue())
                self.assertNotIn('fundamentals_report', state)
                self.assertEqual(len(tools[NAMES[0]].calls), int(after_data))

    def test_no_source_does_not_allow_unregistered_tool(self):
        node, state, model, tools = self.fixture([Message(tool_calls=[call()])], openai=False)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'FUNDAMENTALS_UNKNOWN_TOOL'):
            node(state)
        self.assertTrue(all(not tool.calls for tool in tools.values()))

    def test_empty_narrative_remains_explicit(self):
        node, state, model, tools = self.fixture([Message()])
        self.assertIn('did not return a complete fundamentals narrative', node(state)['fundamentals_report'])

    def test_tool_exception_text_is_not_forwarded(self):
        node, state, model, tools = self.fixture([Message(tool_calls=[call()])])
        tools[NAMES[0]].fail = True
        node(state)
        self.assertIn('RuntimeError', model.histories[1][1].content)
        self.assertNotIn('private-credential-fixture', model.histories[1][1].content)

    def test_source_selection_unchanged(self):
        cases = [({'openai': False}, []),
            ({'openai': False, 'finnhub': True}, list(NAMES[1:3])),
            ({'openai': False, 'simfin': True}, list(NAMES[3:6])),
            ({'finnhub': True, 'simfin': True}, list(NAMES[:6])),
            ({'ticker': 'BTC/USD', 'finnhub': True, 'simfin': True}, [NAMES[6], NAMES[0]])]
        for options, expected in cases:
            with self.subTest(options=options):
                node, state, model, tools = self.fixture([Message(content='FINAL TRANSACTION PROPOSAL: HOLD')], **options)
                node(state)
                self.assertEqual(self.bound_names, expected)

    def test_text_blocks_and_argument_copy(self):
        self.assertEqual(self.ns['_fundamentals_text']([
            {'type': 'text', 'text': 'First'}, {'type': 'text', 'text': 'Second'},
            {'type': 'reasoning', 'text': 'Hidden'}]), 'First\nSecond')
        message = Message(tool_calls=[call()])
        parsed = self.ns['_fundamentals_tool_calls'](message)
        parsed[0]['args']['ticker'] = 'CHANGED'
        self.assertEqual(message.tool_calls[0]['args']['ticker'], 'AAPL')

    def test_loop_limit_remains_explicit(self):
        node, state, model, tools = self.fixture(
            [Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])], iterations=1)
        self.assertIn('Tool-loop halted after 1 iterations', node(state)['fundamentals_report'])
        self.assertEqual(len(tools[NAMES[0]].calls), 1)
        self.assertEqual(len(model.histories), 3)


if __name__ == '__main__':
    unittest.main()
