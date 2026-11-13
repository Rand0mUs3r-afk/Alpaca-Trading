"""Paper-only research runner with optional Yahoo/AI research-ticker selection.
Check mode never selects or researches. Automatic selection is analyze-only.
Receipt checks establish provenance/completeness, not truth of every AI claim.
Offline tests: python -m scripts.paper_trading_runner --self-test
"""
import argparse
import copy
import functools
import html
import importlib
import json
from pathlib import Path
from types import SimpleNamespace


def checked_llm(llm, quality):
    cls = type(llm)
    if not cls.__dict__.get('__paper_quality_checked__', False):
        original = cls._generate
        @functools.wraps(original)
        def generate(self, *args, **kwargs):
            return quality.normalized_chat_result(original(self, *args, **kwargs))
        cls._generate = generate
        cls.__paper_quality_checked__ = True
    return llm


def install_reply_checks(free, quality):
    original = free.build_llm
    def build(provider, *args, **kwargs):
        llm = original(provider, *args, **kwargs)
        return checked_llm(llm, quality) if provider == 'groq' else llm
    free.build_llm = build


def selection_transport(quota, original):
    def one_request(send, client, request, args, kwargs, budget, **options):
        def once(client, request, *args, **kwargs):
            response = send(client, request, *args, **kwargs)
            if response.status_code in (413, 429):
                reason, info = quota.quota_reason(response)
                budget.update(response.headers)
                response.close()
                raise quota.QuotaStop(reason, info)
            return response
        return original(once, client, request, args, kwargs, budget, **options)
    return one_request


def selection_reply(free, quota, provider, model, messages, max_output_tokens=1024):
    """One SDK call, no selection retries/fallback; shared HTTP quota guard applies."""
    import time
    from langchain_openai import ChatOpenAI
    from tradingagents.llm_clients.google_client import NormalizedChatGoogleGenerativeAI
    class GoogleSelection(NormalizedChatGoogleGenerativeAI):
        def __init__(self, **kwargs):
            kwargs.update(max_retries=0, max_output_tokens=max_output_tokens, temperature=0)
            super().__init__(**kwargs)
    class GroqSelection(ChatOpenAI):
        def __init__(self, **kwargs):
            kwargs.update(max_retries=0, max_tokens=max_output_tokens, temperature=0)
            super().__init__(**kwargs)
    llm = free.build_llm(provider, model, GoogleSelection, GroqSelection)
    # Use shared pacing but NOT paced(), which may retry transient errors.
    with free.LOCK:
        time.sleep(max(0, 20 - (time.monotonic() - free.LAST_CALL)))
        free.LAST_CALL = time.monotonic()
        original_send = quota.send_with_limits
        quota.send_with_limits = selection_transport(quota, original_send)
        try:
            # The installed builder resets Groq max_tokens; override it per invocation.
            reply = llm.invoke(messages, **({'max_tokens': max_output_tokens} if provider == 'groq' else {}))
        finally:
            quota.send_with_limits = original_send
    content = reply.content
    if isinstance(content, list):
        content = ''.join(item.get('text', '') for item in content
                          if isinstance(item, dict) and item.get('type') == 'text')
    return content


def select_research_ticker(d, choices, result):
    free = d.free_providers
    selector = getattr(d, 'yahoo_ticker_selection', None)
    if selector is None:
        selector = importlib.import_module('scripts.yahoo_ticker_selection')
    free.PHASE = 'yahoo_candidate_fetch'
    result['ticker_selection'] = {'status': 'FETCHING', 'method': 'yahoo_most_active_ai'}
    snapshot = selector.fetch_snapshot()
    provider, model = choices[0]
    free.ACTIVE.update(provider=provider, model=model)
    result['ticker_selection'].update(status='SELECTING', source_url=snapshot['source_url'],
                                     fetched_at_utc=snapshot['fetched_at_utc'], provider=provider, model=model)
    free.PHASE = 'ai_ticker_selection'
    ask = getattr(d, 'selection_ask', None)
    if ask is None:
        ask = lambda messages, **kwargs: selection_reply(free, d.free_groq_limits, provider, model, messages, **kwargs)
    selected = selector.select_ticker(snapshot, ask)
    free.PHASE = 'selected_ticker_validation'
    d.free_actions.validate(selected['ticker'], 'check', result['requested_model'], False)
    result['ticker_selection'].update(selected, status='SELECTED')
    result['symbol'] = selected['ticker']
    print('RESEARCH_TICKER_SELECTED: ' + selected['ticker'], flush=True)
    return selected['ticker']


def run_with_receipts(free, app, provider, model, symbol):
    saved = (app.tool_calls_log, app.tool_calls_count, app.analyzing_symbol, app.current_symbol)
    app.tool_calls_log, app.tool_calls_count = [], 0
    app.analyzing_symbol = app.current_symbol = symbol
    try:
        graph = free.run_graph(provider, model, free._paper_graph_class)
        state, signal = graph.propagate(symbol, free._paper_trade_date())
        receipts = [copy.deepcopy(item) for item in app.tool_calls_log if isinstance(item, dict)]
        return state, signal, receipts
    finally:
        app.tool_calls_log, app.tool_calls_count, app.analyzing_symbol, app.current_symbol = saved


def prepare(quality, state, signal, receipts, symbol):
    payload = quality.execution_payload(state, signal, receipts, symbol)
    gate = payload['evidence_quality']
    if gate['reports'].get('macro') == 'unverified' and quality.usable_report(payload['reports'].get('macro_report')):
        global_tools = []
        for receipt in receipts:
            inputs = receipt.get('inputs') or {}
            if (receipt.get('symbol') == symbol and receipt.get('tool_name') in quality.TOOLS['macro']
                    and not (inputs.get('ticker') or inputs.get('symbol') or inputs.get('query'))
                    and (not inputs.get('ticker_context') or str(inputs['ticker_context']).upper() == symbol)):
                view = {**receipt, 'inputs': {**inputs, 'ticker': symbol}}
                if quality.receipt_ok(view, 'macro', symbol):
                    global_tools.append(receipt['tool_name'])
        if global_tools:
            gate['reports']['macro'] = 'verified'
            gate['accepted_tools']['macro'] = sorted(set(global_tools))
            gate['reasons'] = [reason for reason in gate['reasons'] if reason != 'UNVERIFIED_MACRO']
            gate['passed'] = not gate['reasons']
            gate['scope_notes'] = {'macro': 'Global inputs matched by application attempt context; ticker is not a macro API parameter.'}
            payload['status'] = 'ANALYSIS_COMPLETE_NO_ORDERS' if gate['passed'] else 'ANALYSIS_INCOMPLETE'
            payload['signal'] = payload['model_signal'] if gate['passed'] else 'NO_TRADE'
    payload['tool_receipt_count'] = len(receipts)
    return payload


def execute_if_requested(payload, engine, broker, requested):
    if not requested:
        return {'status': 'DISABLED', 'submitted': False}
    if not payload['evidence_quality']['passed']:
        return {'status': 'BLOCKED_EVIDENCE', 'submitted': False,
                'reasons': payload['evidence_quality']['reasons']}
    try:
        return engine.execute(payload, broker, enabled=True)
    except engine.Blocked as exc:
        return {'status': 'BLOCKED', 'submitted': False, 'reason': str(exc)}
    except Exception as exc:
        return {'status': 'SUBMISSION_UNKNOWN', 'submitted': None,
                'error_type': type(exc).__name__,
                'help': 'Inspect paper orders before retrying. No automatic resubmission.'}


def dependencies():
    names = ('free_providers', 'free_actions', 'free_schema_compat', 'free_compact_context',
             'free_request_fit', 'free_groq_limits', 'analysis_quality', 'paper_executor', 'yahoo_ticker_selection')
    modules = {name: importlib.import_module('scripts.' + name) for name in names}
    modules['app'] = importlib.import_module('webui.utils.state').app_state
    return SimpleNamespace(**modules)


def main(argv=None, deps=None):
    import os
    import re
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'analyze', 'paper'), default='check')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--ticker-source', choices=('manual', 'yahoo_most_active'), default='manual')
    parser.add_argument('--provider', choices=('auto', 'gemini', 'groq'), default='auto')
    parser.add_argument('--model', choices=('gemini-3.8-flash', 'gemini-3.6-flash'), default='gemini-3.8-flash')
    parser.add_argument('--free-tier-confirmed', choices=('true', 'false'), default='false')
    parser.add_argument('--paper-orders-confirmed', choices=('true', 'false'), default='false')
    args = parser.parse_args(argv)
    d = deps or dependencies()
    free, base, quality, engine = d.free_providers, d.free_actions, d.analysis_quality, d.paper_executor
    automatic = args.ticker_source == 'yahoo_most_active'
    symbol = None if automatic else args.ticker
    result = {'status': 'FAILED', 'mode': args.mode, 'symbol': symbol,
              'ticker_source': args.ticker_source, 'requested_model': args.model,
              'requested_provider': args.provider, 'paper_execution_requested': args.mode == 'paper',
              'orders_enabled': False}
    free.ATTEMPTS.clear()
    free.ACTIVE.clear()
    free.PHASE = 'input_validation'
    code = 1
    try:
        # Legacy validator also checks environment policy; sentinel is never researched.
        base.validate('AAPL' if automatic else symbol, 'check', args.model, False)
        if automatic and args.mode == 'paper':
            raise ValueError('AUTO_SELECTION_ANALYZE_ONLY')
        if args.mode == 'paper' and args.paper_orders_confirmed != 'true':
            raise ValueError('PAPER_ORDER_CONFIRMATION_REQUIRED')
        if args.mode != 'check':
            if args.free_tier_confirmed != 'true':
                raise ValueError('FREE_PLAN_CONFIRMATION_REQUIRED')
            if '@' not in os.getenv('SEC_USER_AGENT', ''):
                raise ValueError('SEC_CONTACT_REQUIRED')
            choices = free.routes(args.provider, args.model)
        d.free_schema_compat.install(free, d.free_compact_context, d.free_request_fit, d.free_groq_limits)
        install_reply_checks(free, quality)
        free.PHASE = 'network_guard'
        free.install_http_guard()
        free.PHASE = 'paper_account_check'
        result.update(base.paper_check())
        broker = None
        if args.mode == 'paper':
            broker = engine.PaperBroker()
            engine.clock_ok(broker.clock(), datetime.now(timezone.utc))
        if args.mode == 'check':
            result['status'], code = 'PAPER_CHECK_OK', 0
            if automatic:
                result['ticker_selection'] = {'status': 'SKIPPED_CHECK_MODE'}
        else:
            if automatic:
                symbol = select_research_ticker(d, choices, result)
            import_config = d.__dict__.get('clear_runtime_keys')
            if import_config:
                import_config()
            elif deps is None:
                importlib.import_module('tradingagents.dataflows.config').clear_runtime_api_keys()
            free.PHASE = 'free_research_tools'
            base.install_free_tools()
            free.PHASE = 'local_embeddings'
            base.install_local_memory()
            free.PHASE = 'client_setup'
            free._paper_graph_class = free.install_clients()
            free._paper_trade_date = lambda: datetime.now(ZoneInfo('America/New_York')).date().isoformat()
            state, signal, receipts = free.choose_analysis(choices, lambda provider, model:
                run_with_receipts(free, d.app, provider, model, symbol))
            payload = prepare(quality, state, signal, receipts, symbol)
            result.update(payload)
            requested = args.mode == 'paper'
            outcome = execute_if_requested(payload, engine, broker, requested)
            result['execution'] = outcome
            result['orders_enabled'] = requested and payload['evidence_quality']['passed']
            if not payload['evidence_quality']['passed']:
                result['status'], code = 'ANALYSIS_INCOMPLETE', 2
            elif not requested:
                code = 0
            elif outcome['status'] in ('ORDER_ACCEPTED_NOT_FILL_VERIFIED', 'RECONCILED_ORDER_FOUND'):
                result['status'], code = 'PAPER_ORDER_SUBMITTED_FILL_UNVERIFIED', 0
            elif outcome['status'] in ('NO_ACTION', 'EXISTING_ORDER'):
                result['status'], code = 'PAPER_NO_NEW_ORDER', 0
            else:
                result['status'], code = 'PAPER_EXECUTION_NOT_COMPLETED', 3
    except Exception as exc:
        result.update(free.error_details(exc))
        if isinstance(exc, engine.Blocked):
            result['error_code'] = str(exc)
        elif isinstance(exc, getattr(engine, 'BrokerError', ())):
            result.update(error_code='PAPER_BROKER_HTTP_' + str(exc.status), broker_http_status=exc.status)
        elif str(exc) in ('PAPER_ORDER_CONFIRMATION_REQUIRED', 'FREE_PLAN_CONFIRMATION_REQUIRED',
                          'SEC_CONTACT_REQUIRED', 'AUTO_SELECTION_ANALYZE_ONLY'):
            result['error_code'] = str(exc)
        elif free.PHASE in ('yahoo_candidate_fetch', 'ai_ticker_selection', 'selected_ticker_validation'):
            selector = getattr(d, 'yahoo_ticker_selection', None)
            if selector is not None and isinstance(exc, selector.SelectionError) and re.fullmatch(r'(?:YAHOO|AI_TICKER)_[A-Z0-9_]{1,70}', str(exc)):
                result['error_code'] = str(exc)
        current, seen = exc, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, quality.InvalidModelReply):
                result['error_code'] = current.error_code
                break
            current = current.__cause__ or current.__context__
        if result.get('ticker_selection', {}).get('status') in ('FETCHING', 'SELECTING'):
            result['ticker_selection']['status'] = 'FAILED'
        result['help'] = 'Use error_code/evidence_quality/execution status. Do not share keys. No automatic order retry.'
    result.update(free.ACTIVE)
    result['provider_attempts'] = free.ATTEMPTS
    result['execution_scope'] = 'New paper long entries only; SELL/HOLD do not close positions.'
    result['quality_scope'] = 'Report completeness and matching tool receipts, not verification of every assertion.'
    base.REPORT.parent.mkdir(parents=True, exist_ok=True)
    text = base.redact(json.dumps(result, indent=2, ensure_ascii=True))
    base.REPORT.write_text(text + '\n', encoding='utf-8')
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as handle:
            handle.write('# Paper trading integration\n\n<pre>' + html.escape(text) + '</pre>\n')
    print(result['status'])
    fields = ('status', 'symbol', 'ticker_source', 'error_code', 'error_type', 'error_stage',
              'http_status', 'provider', 'model', 'provider_error_code', 'provider_error_hints', 'rate_limits')
    print('RUN_RESULT_SUMMARY: ' + base.redact(json.dumps({key: result[key] for key in fields if key in result})), flush=True)
    return code


def self_test():
    import os
    import tempfile
    import unittest
    from unittest.mock import Mock, patch
    class Blocked(ValueError):
        pass
    class InvalidReply(ValueError):
        pass
    class SelectionError(ValueError):
        pass
    def fixture(directory, passed=True, signal='BUY', broker_error=False, selection_error=False):
        calls = []
        app = SimpleNamespace(tool_calls_log=[{'old': True}], tool_calls_count=1,
                              analyzing_symbol='OLD', current_symbol='OLD')
        free = SimpleNamespace(ATTEMPTS=[], ACTIVE={}, PHASE='startup', build_llm=lambda *a, **k: None,
            routes=lambda *a: [('groq', 'openai/gpt-oss-120b')], install_http_guard=lambda: None,
            install_clients=lambda: object,
            error_details=lambda exc: {'error_type': type(exc).__name__, 'error_stage': free.PHASE})
        def propagate(symbol, date):
            calls.append(('analysis', symbol))
            app.tool_calls_log.append({'symbol': symbol, 'status': 'success'})
            return {'final_trade_intent': {'action': signal}}, signal
        free.run_graph = lambda *a: SimpleNamespace(propagate=propagate)
        def choose(choices, run):
            free.ACTIVE.update(provider=choices[0][0], model=choices[0][1])
            return run(*choices[0])
        free.choose_analysis = choose
        def payload(state, action, receipts, symbol):
            return {'status': 'ANALYSIS_COMPLETE_NO_ORDERS' if passed else 'ANALYSIS_INCOMPLETE',
                'symbol': symbol, 'signal': action if passed else 'NO_TRADE', 'model_signal': action,
                'reports': {'macro_report': ''}, 'trade_intent': state['final_trade_intent'],
                'evidence_quality': {'passed': passed, 'reports': {'macro': 'unavailable'},
                                    'reasons': [] if passed else ['UNVERIFIED_MARKET'], 'accepted_tools': {}}}
        quality = SimpleNamespace(execution_payload=payload, InvalidModelReply=InvalidReply,
                                  usable_report=lambda value: False, TOOLS={'macro': set()})
        def execute(value, broker, enabled=False):
            calls.append(('execute', enabled))
            if broker_error:
                raise RuntimeError('private-broker-value')
            return {'status': 'NO_ACTION' if signal != 'BUY' else 'ORDER_ACCEPTED_NOT_FILL_VERIFIED',
                    'submitted': signal == 'BUY'}
        engine = SimpleNamespace(Blocked=Blocked, execute=execute, clock_ok=lambda *a: None,
                                 PaperBroker=lambda: SimpleNamespace(clock=lambda: {'is_open': True}))
        base = SimpleNamespace(validate=lambda *a: None, paper_check=lambda: {'paper_account_verified': True},
            install_free_tools=lambda: None, install_local_memory=lambda: None,
            REPORT=Path(directory)/'result.json', redact=lambda value: value)
        def select(snapshot, ask):
            if selection_error:
                raise SelectionError('AI_TICKER_NOT_IN_CANDIDATES')
            answer = json.loads(ask([], max_output_tokens=1024))
            return dict(answer, source_url=snapshot['source_url'], candidates_considered=[{'ticker': 'MSFT'}])
        selector = SimpleNamespace(SelectionError=SelectionError, select_ticker=select,
            fetch_snapshot=Mock(return_value={'source_url': 'https://finance.yahoo.com/markets/stocks/most-active/',
                                             'fetched_at_utc': 'fixture'}))
        deps = SimpleNamespace(free_providers=free, free_actions=base, analysis_quality=quality, paper_executor=engine,
            free_schema_compat=SimpleNamespace(install=lambda *a: None), free_compact_context=None,
            free_request_fit=None, free_groq_limits=None, app=app, clear_runtime_keys=lambda: None,
            yahoo_ticker_selection=selector, selection_ask=Mock(return_value='{"ticker":"MSFT","reason":"Research."}'))
        return deps, calls
    class Tests(unittest.TestCase):
        def run_case(self, argv, **options):
            with tempfile.TemporaryDirectory() as directory:
                deps, calls = fixture(directory, **options)
                with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test test@example.com'}, clear=True):
                    code = main(argv, deps)
                return code, json.loads(deps.free_actions.REPORT.read_text()), calls, deps
        def test_selection_transport_stops_after_first_quota_response(self):
            class Stop(RuntimeError):
                def __init__(self, code, details): super().__init__(code)
            quota = SimpleNamespace(QuotaStop=Stop, quota_reason=lambda response: ('LIMIT', {}))
            for status in (413, 429):
                response = Mock(status_code=status, headers={})
                send, budget = Mock(return_value=response), Mock()
                def retrying(send, client, request, args, kwargs, budget, **options):
                    for _ in range(3): send(client, request)
                with self.assertRaises(Stop):
                    selection_transport(quota, retrying)(send, None, None, (), {}, budget)
                send.assert_called_once()
                response.close.assert_called_once()
        def test_selection_transport_success_is_unchanged(self):
            response = Mock(status_code=200)
            send = Mock(return_value=response)
            original = lambda send, client, request, args, kwargs, budget, **options: send(client, request)
            result = selection_transport(SimpleNamespace(), original)(send, None, None, (), {}, Mock())
            self.assertIs(result, response)
            send.assert_called_once()
        def test_check_has_no_analysis_or_orders(self):
            code, report, calls, deps = self.run_case(['--mode', 'check'])
            self.assertEqual((code, calls), (0, []))
            deps.yahoo_ticker_selection.fetch_snapshot.assert_not_called()
        def test_auto_check_has_no_yahoo_or_ai(self):
            code, report, calls, deps = self.run_case(['--ticker-source', 'yahoo_most_active'])
            self.assertEqual(code, 0)
            self.assertIsNone(report['symbol'])
            deps.yahoo_ticker_selection.fetch_snapshot.assert_not_called()
            deps.selection_ask.assert_not_called()
        def test_manual_analysis_preserved(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--ticker', 'TSLA', '--free-tier-confirmed', 'true'])
            self.assertEqual(code, 0)
            self.assertIn(('analysis', 'TSLA'), calls)
            self.assertNotIn(('execute', True), calls)
            deps.selection_ask.assert_not_called()
        def test_auto_selected_symbol_reaches_research_and_report(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--ticker-source', 'yahoo_most_active', '--free-tier-confirmed', 'true'])
            self.assertEqual((code, report['symbol']), (0, 'MSFT'))
            self.assertEqual(calls, [('analysis', 'MSFT')])
            self.assertFalse(report['orders_enabled'])
            self.assertEqual(report['ticker_selection']['status'], 'SELECTED')
            deps.selection_ask.assert_called_once()
        def test_auto_failure_stops_before_research(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--ticker-source', 'yahoo_most_active', '--free-tier-confirmed', 'true'], selection_error=True)
            self.assertEqual((code, calls), (1, []))
            self.assertEqual(report['error_code'], 'AI_TICKER_NOT_IN_CANDIDATES')
            self.assertIsNone(report['symbol'])
        def test_auto_paper_is_rejected_even_if_confirmed(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--ticker-source', 'yahoo_most_active', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'])
            self.assertEqual(report['error_code'], 'AUTO_SELECTION_ANALYZE_ONLY')
            self.assertEqual(calls, [])
            deps.yahoo_ticker_selection.fetch_snapshot.assert_not_called()
        def test_free_confirmation_before_selection(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--ticker-source', 'yahoo_most_active'])
            self.assertEqual(report['error_code'], 'FREE_PLAN_CONFIRMATION_REQUIRED')
            deps.selection_ask.assert_not_called()
        def test_manual_paper_requires_confirmation(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true'])
            self.assertEqual(report['error_code'], 'PAPER_ORDER_CONFIRMATION_REQUIRED')
            self.assertEqual(calls, [])
        def test_manual_paper_evidence_failure_blocks_execution(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], passed=False)
            self.assertEqual(code, 2)
            self.assertEqual(report['execution']['status'], 'BLOCKED_EVIDENCE')
            self.assertNotIn(('execute', True), calls)
        def test_manual_confirmed_paper_executes_once(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'])
            self.assertEqual(code, 0)
            self.assertEqual(calls.count(('execute', True)), 1)
        def test_unknown_submission_is_not_retried_or_leaked(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], broker_error=True)
            self.assertEqual(code, 3)
            self.assertEqual(calls.count(('execute', True)), 1)
            self.assertEqual(calls.count(('analysis', 'AAPL')), 1)
            self.assertNotIn('private-broker-value', json.dumps(report))
        def test_hold_is_not_an_order(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], signal='HOLD')
            self.assertEqual(report['status'], 'PAPER_NO_NEW_ORDER')
            self.assertFalse(report['execution']['submitted'])
        def test_app_state_restored(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--free-tier-confirmed', 'true'])
            self.assertEqual(deps.app.tool_calls_log, [{'old': True}])
            self.assertEqual(deps.app.current_symbol, 'OLD')
            self.assertEqual(report['tool_receipt_count'], 1)
        def test_reply_normalization_only_once(self):
            class LLM:
                def _generate(self): return 'raw'
            q = SimpleNamespace(normalized_chat_result=lambda result: 'checked-' + result)
            llm = LLM()
            checked_llm(llm, q)
            checked_llm(llm, q)
            self.assertEqual(llm._generate(), 'checked-raw')
        def test_failed_graph_restores_receipts(self):
            with tempfile.TemporaryDirectory() as directory:
                deps, calls = fixture(directory)
                def fail(*args):
                    deps.app.tool_calls_log.append({'status': 'error'})
                    raise RuntimeError('fixture')
                free = deps.free_providers
                free._paper_graph_class, free._paper_trade_date = object, lambda: 'fixture'
                free.run_graph = lambda *a: SimpleNamespace(propagate=fail)
                with self.assertRaises(RuntimeError): run_with_receipts(free, deps.app, 'groq', 'model', 'AAPL')
                self.assertEqual(deps.app.tool_calls_log, [{'old': True}])
        def test_macro_scope_cannot_override_other_failures(self):
            for extra, receipt_symbol in ((False, 'AAPL'), (True, 'AAPL'), (False, 'OTHER')):
                gate = {'passed': False, 'reports': {'macro': 'unverified'},
                        'reasons': ['UNVERIFIED_MACRO'] + (['UNVERIFIED_MARKET'] if extra else []), 'accepted_tools': {}}
                data = {'evidence_quality': gate, 'reports': {'macro_report': 'source'}, 'model_signal': 'BUY', 'signal': 'NO_TRADE'}
                q = SimpleNamespace(execution_payload=lambda *a: copy.deepcopy(data), usable_report=lambda value: True,
                    TOOLS={'macro': {'get_macro_analysis'}}, receipt_ok=lambda *a: True)
                receipt = {'symbol': receipt_symbol, 'tool_name': 'get_macro_analysis', 'inputs': {}}
                saved = copy.deepcopy(receipt)
                result = prepare(q, {}, 'BUY', [receipt], 'AAPL')
                self.assertEqual(result['evidence_quality']['passed'], not extra and receipt_symbol == 'AAPL')
                self.assertEqual(receipt, saved)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(0 if self_test() else 1)
    raise SystemExit(main())
