"""Offline social protocol regressions, runnable before optional SDK installation.
Executes the production functions with stand-in messages, prompts and tools.
This does not validate real provider behavior or SDK serialization.
"""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class Message:
    def __init__(self, content='', tool_calls=None, additional_kwargs=None, tool_call_id=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = additional_kwargs or {}
        self.tool_call_id = tool_call_id
        self.invalid_tool_calls = []
        self.response_metadata = {'fixture': 'preserve-me'}


class Prompt:
    @classmethod
    def from_messages(cls, *args):
        return cls()
    def partial(self, **kwargs):
        return self
    def format_messages(self, **kwargs):
        return [Message(content='Resolved fixture prompt')]
    def __or__(self, model):
        return model


class Tool:
    def __init__(self, name):
        self.name, self.calls, self.fail = name, [], False
    def invoke(self, args):
        self.calls.append(copy.deepcopy(args))
        if self.fail:
            raise RuntimeError('private-credential-fixture')
        return 'Fixture source observations'


class Model:
    def __init__(self, replies):
        self.replies, self.histories = list(replies), []
    def bind_tools(self, tools):
        return self
    def invoke(self, history):
        self.histories.append(list(history) if isinstance(history, list) else history)
        if not self.replies:
            raise AssertionError('Unexpected extra model call')
        return self.replies.pop(0)


def call(call_id='call_1', name='get_reddit_stock_info'):
    return {'id': call_id, 'name': name, 'args': {'ticker': 'AAPL', 'curr_date': '2026-11-12'}}


class SocialProtocolTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'tradingagents/agents/analysts/social_media_analyst.py'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        names = {'_social_tool_calls', '_social_text', 'create_social_media_analyst'}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in selected}, names)
        self.ns = {'json': json, 'AIMessage': Message, 'ToolMessage': Message,
            'ChatPromptTemplate': Prompt, 'MessagesPlaceholder': lambda **kwargs: None,
            'load_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'render_prompt': lambda *args, **kwargs: 'Fixture prompt',
            'capture_agent_prompt': lambda *args, **kwargs: None}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), self.ns)

    def fixture(self, replies, iterations=8):
        reddit, news = Tool('get_reddit_stock_info'), Tool('get_stock_news_openai')
        toolkit = SimpleNamespace(config={'online_tools': True, 'max_tool_iterations_per_agent': iterations},
            has_openai_web_search=lambda: True, get_reddit_stock_info=reddit, get_stock_news_openai=news)
        model = Model(replies + [Message(content='Fixture report. FINAL TRANSACTION PROPOSAL: HOLD')])
        node = self.ns['create_social_media_analyst'](model, toolkit)
        state = {'trade_date': '2026-11-12', 'company_of_interest': 'AAPL', 'messages': []}
        return node, state, model, reddit, news

    def test_canonical_calls(self):
        self.assertEqual(self.ns['_social_tool_calls'](Message(tool_calls=[call()])), [call()])

    def test_legacy_calls(self):
        raw = {'id': 'call_1', 'type': 'function', 'function': {'name': call()['name'], 'arguments': json.dumps(call()['args'])}}
        self.assertEqual(self.ns['_social_tool_calls'](Message(additional_kwargs={'tool_calls': [raw]})), [call()])

    def test_canonical_calls_take_precedence(self):
        message = Message(tool_calls=[call()], additional_kwargs={'tool_calls': ['invalid-unused-legacy']})
        self.assertEqual(self.ns['_social_tool_calls'](message), [call()])

    def test_bad_arguments_rejected(self):
        for args in ('not-json', [], None, {'price': float('nan')}):
            with self.subTest(args_type=type(args).__name__):
                with self.assertRaisesRegex(ValueError, 'SOCIAL_INVALID_TOOL_ARGUMENTS'):
                    self.ns['_social_tool_calls'](Message(tool_calls=[{**call(), 'args': args}]))

    def test_missing_ids_rejected(self):
        for value in (None, '', 12):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'SOCIAL_INVALID_TOOL_NAME_OR_ID'):
                    self.ns['_social_tool_calls'](Message(tool_calls=[{**call(), 'id': value}]))

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SOCIAL_DUPLICATE_TOOL_CALL_ID'):
            self.ns['_social_tool_calls'](Message(tool_calls=[call(), call()]))

    def test_invalid_sdk_calls_rejected(self):
        message = Message()
        message.invalid_tool_calls = [{'error': 'Fixture malformed call'}]
        with self.assertRaisesRegex(ValueError, 'SOCIAL_INVALID_TOOL_CALL'):
            self.ns['_social_tool_calls'](message)

    def test_argument_copy_protects_original_message(self):
        message = Message(tool_calls=[call()])
        parsed = self.ns['_social_tool_calls'](message)
        parsed[0]['args']['ticker'] = 'CHANGED'
        self.assertEqual(message.tool_calls[0]['args']['ticker'], 'AAPL')

    def test_null_legacy_calls_mean_no_calls(self):
        self.assertEqual(self.ns['_social_tool_calls'](Message(additional_kwargs={'tool_calls': None})), [])

    def test_only_visible_text_blocks_are_used(self):
        blocks = [{'type': 'text', 'text': 'Visible'}, {'type': 'reasoning', 'text': 'Not report text'}]
        self.assertEqual(self.ns['_social_text'](blocks), 'Visible')

    def test_original_ai_message_is_preserved_once(self):
        first = Message(content='Original tool-request text', tool_calls=[call()])
        node, state, model, reddit, news = self.fixture([first])
        node(state)
        self.assertIs(model.histories[1][0], first)
        self.assertEqual(model.histories[1][0].response_metadata, {'fixture': 'preserve-me'})
        self.assertEqual(model.histories[1][1].tool_call_id, 'call_1')
        self.assertEqual(state['messages'], [])

    def test_multiple_tools_share_one_original_ai_message(self):
        first = Message(tool_calls=[call(), call('call_2', 'get_stock_news_openai')])
        node, state, model, reddit, news = self.fixture([first])
        node(state)
        self.assertEqual(len(model.histories[1]), 3)
        self.assertIs(model.histories[1][0], first)
        self.assertEqual([item.tool_call_id for item in model.histories[1][1:]], ['call_1', 'call_2'])
        self.assertEqual((len(reddit.calls), len(news.calls)), (1, 1))

    def test_identical_calls_use_cached_results(self):
        node, state, model, reddit, news = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])])
        node(state)
        self.assertEqual(len(reddit.calls), 1)
        self.assertEqual(len(model.histories), 3)

    def test_reused_ids_across_rounds_are_rejected(self):
        node, state, model, reddit, news = self.fixture([Message(tool_calls=[call()]), Message(tool_calls=[call()])])
        with self.assertRaisesRegex(ValueError, 'SOCIAL_REUSED_TOOL_CALL_ID'):
            node(state)
        self.assertEqual(len(reddit.calls), 1)

    def test_unknown_tool_blocks_whole_batch_before_execution(self):
        first = Message(tool_calls=[call(), call('call_2', 'unknown_tool')])
        node, state, model, reddit, news = self.fixture([first])
        with self.assertRaisesRegex(ValueError, 'SOCIAL_UNKNOWN_TOOL'):
            node(state)
        self.assertEqual(reddit.calls, [])
        self.assertEqual(news.calls, [])

    def test_tool_errors_do_not_echo_private_exception_text(self):
        node, state, model, reddit, news = self.fixture([Message(tool_calls=[call()])])
        reddit.fail = True
        node(state)
        result = model.histories[1][1].content
        self.assertIn('RuntimeError', result)
        self.assertNotIn('private-credential-fixture', result)

    def test_tool_loop_limit_is_explicit(self):
        node, state, model, reddit, news = self.fixture(
            [Message(tool_calls=[call()]), Message(tool_calls=[call('call_2')])], iterations=1)
        result = node(state)
        self.assertIn('Tool-loop halted after 1 iterations', result['sentiment_report'])
        self.assertEqual(len(reddit.calls), 1)
        self.assertEqual(len(model.histories), 3)


if __name__ == '__main__':
    unittest.main()
