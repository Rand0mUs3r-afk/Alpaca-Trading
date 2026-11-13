"""Step 3: connect free analysis, tool evidence and the paper-only executor.
Actions still needs a separate workflow edit to use this launcher.
Modes: check, analyze, paper. Paper additionally requires --paper-orders-confirmed true.
No live trading, scheduling or SELL-position management. New long entries only.
Receipt checks are provenance/completeness checks, not proof of every model claim.
Offline self-tests: python -m scripts.paper_trading_runner --self-test
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
        # Google's adapter already performs its own message normalization.
        return checked_llm(llm, quality) if provider == 'groq' else llm
    free.build_llm = build


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
    # Global macro APIs do not take a stock ticker. Attribute only actual receipts
    # scoped by this application's current symbol; never alter the stored receipt.
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
        # Blocked messages in paper_executor are fixed codes, not provider bodies.
        return {'status': 'BLOCKED', 'submitted': False, 'reason': str(exc)}
    except Exception as exc:
        # Execution is outside the provider-fallback loop. Never rerun analysis or
        # submit another order automatically after an ambiguous broker exception.
        return {'status': 'SUBMISSION_UNKNOWN', 'submitted': None,
                'error_type': type(exc).__name__,
                'help': 'Inspect paper orders before retrying. No automatic resubmission.'}


def dependencies():
    names = ('free_providers', 'free_actions', 'free_schema_compat', 'free_compact_context',
             'free_request_fit', 'free_groq_limits', 'analysis_quality', 'paper_executor')
    modules = {name: importlib.import_module('scripts.' + name) for name in names}
    modules['app'] = importlib.import_module('webui.utils.state').app_state
    return SimpleNamespace(**modules)


def main(argv=None, deps=None):
    import os
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'analyze', 'paper'), default='check')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--provider', choices=('auto', 'gemini', 'groq'), default='auto')
    parser.add_argument('--model', choices=('gemini-3.8-flash', 'gemini-3.6-flash'), default='gemini-3.8-flash')
    parser.add_argument('--free-tier-confirmed', choices=('true', 'false'), default='false')
    parser.add_argument('--paper-orders-confirmed', choices=('true', 'false'), default='false')
    args = parser.parse_args(argv)
    d = deps or dependencies()
    free, base, quality, engine = d.free_providers, d.free_actions, d.analysis_quality, d.paper_executor
    result = {'status': 'FAILED', 'mode': args.mode, 'symbol': args.ticker,
              'requested_provider': args.provider, 'paper_execution_requested': args.mode == 'paper',
              'orders_enabled': False}
    free.ATTEMPTS.clear()
    free.ACTIVE.clear()
    free.PHASE = 'input_validation'
    code = 1
    try:
        base.validate(args.ticker, 'check', args.model, False)
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
        else:
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
                run_with_receipts(free, d.app, provider, model, args.ticker))
            payload = prepare(quality, state, signal, receipts, args.ticker)
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
        elif str(exc) in ('PAPER_ORDER_CONFIRMATION_REQUIRED', 'FREE_PLAN_CONFIRMATION_REQUIRED', 'SEC_CONTACT_REQUIRED'):
            result['error_code'] = str(exc)
        current, seen = exc, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, quality.InvalidModelReply):
                result['error_code'] = current.error_code
                break
            current = current.__cause__ or current.__context__
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
    return code


def self_test():
    from types import SimpleNamespace as NS
    import os
    import tempfile
    import unittest
    from unittest.mock import patch
    class Blocked(ValueError):
        pass
    class ModelError(ValueError):
        pass
    def fixture(directory, passed=True, signal='BUY', broker_error=False):
        app = NS(tool_calls_log=[{'old': True}], tool_calls_count=1,
                 analyzing_symbol='OLD', current_symbol='OLD')
        calls = []
        free = NS(ATTEMPTS=[], ACTIVE={}, PHASE='startup', build_llm=lambda *a, **k: None,
                  routes=lambda provider, model: [('groq', 'openai/gpt-oss-120b')],
                  install_http_guard=lambda: None, install_clients=lambda: object,
                  error_details=lambda exc: {'error_type': type(exc).__name__, 'error_stage': 'test'})
        state = {'market_report': 'source report', 'final_trade_intent': {'action': signal}}
        class Graph:
            def propagate(self, symbol, date):
                self_symbol = app.analyzing_symbol
                app.tool_calls_log.append({'symbol': self_symbol, 'status': 'success'})
                calls.append('analysis')
                return state, signal
        free.run_graph = lambda *args: Graph()
        def choose(choices, run):
            provider, model = choices[0]
            free.ACTIVE.update(provider=provider, model=model)
            return run(provider, model)
        free.choose_analysis = choose
        def payload(state, action, receipts, symbol):
            return {'status': 'ANALYSIS_COMPLETE_NO_ORDERS' if passed else 'ANALYSIS_INCOMPLETE',
                'symbol': symbol, 'signal': action if passed else 'NO_TRADE', 'model_signal': action,
                'reports': {'macro_report': ''}, 'trade_intent': state['final_trade_intent'], 'orders_enabled': False,
                'evidence_quality': {'passed': passed, 'reports': {'macro': 'unavailable'},
                                     'reasons': [] if passed else ['UNVERIFIED_MARKET'], 'accepted_tools': {}}}
        quality = NS(execution_payload=payload, InvalidModelReply=ModelError,
                     usable_report=lambda value: False, TOOLS={'macro': set()})
        def execute(value, broker, enabled=False):
            calls.append(('execute', enabled))
            if broker_error:
                raise RuntimeError('private-broker-value')
            if value['signal'] != 'BUY':
                return {'status': 'NO_ACTION', 'submitted': False}
            return {'status': 'ORDER_ACCEPTED_NOT_FILL_VERIFIED', 'submitted': True, 'fill_verified': False}
        engine = NS(Blocked=Blocked, execute=execute,
                    PaperBroker=lambda: NS(clock=lambda: {'is_open': True}), clock_ok=lambda *args: None)
        base = NS(validate=lambda *args: None, paper_check=lambda: {'paper_account_verified': True},
                  install_free_tools=lambda: None, install_local_memory=lambda: None,
                  REPORT=Path(directory)/'result.json', redact=lambda value: value)
        deps = NS(free_providers=free, free_actions=base, analysis_quality=quality, paper_executor=engine,
                  free_schema_compat=NS(install=lambda *args: None), free_compact_context=None,
                  free_request_fit=None, free_groq_limits=None, app=app, clear_runtime_keys=lambda: None)
        return deps, calls
    class Tests(unittest.TestCase):
        def run_case(self, argv, **options):
            with tempfile.TemporaryDirectory() as directory:
                deps, calls = fixture(directory, **options)
                with patch.dict(os.environ, {'SEC_USER_AGENT': 'Test test@example.com'}, clear=True):
                    code = main(argv, deps)
                report = json.loads(deps.free_actions.REPORT.read_text())
                return code, report, calls, deps
        def test_check_never_calls_analysis_or_executor(self):
            code, report, calls, deps = self.run_case(['--mode', 'check'])
            self.assertEqual(code, 0)
            self.assertEqual(calls, [])
        def test_analyze_never_executes(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--free-tier-confirmed', 'true'])
            self.assertEqual(code, 0)
            self.assertEqual(report['execution']['status'], 'DISABLED')
            self.assertNotIn(('execute', True), calls)
        def test_paper_requires_explicit_confirmation(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true'])
            self.assertEqual(code, 1)
            self.assertEqual(calls, [])
        def test_evidence_failure_blocks_even_confirmed_paper_mode(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], passed=False)
            self.assertEqual(code, 2)
            self.assertEqual(report['execution']['status'], 'BLOCKED_EVIDENCE')
            self.assertNotIn(('execute', True), calls)
        def test_confirmed_paper_buy_calls_executor_once(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'])
            self.assertEqual(code, 0)
            self.assertEqual(calls.count(('execute', True)), 1)
            self.assertEqual(report['status'], 'PAPER_ORDER_SUBMITTED_FILL_UNVERIFIED')
        def test_broker_exception_does_not_rerun_analysis_or_leak_message(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], broker_error=True)
            self.assertEqual(code, 3)
            self.assertEqual(calls.count('analysis'), 1)
            self.assertEqual(calls.count(('execute', True)), 1)
            self.assertNotIn('private-broker-value', json.dumps(report))
        def test_tool_log_is_restored_after_attempt(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze', '--free-tier-confirmed', 'true'])
            self.assertEqual(deps.app.tool_calls_log, [{'old': True}])
            self.assertEqual(deps.app.current_symbol, 'OLD')
            self.assertEqual(report['tool_receipt_count'], 1)
        def test_hold_does_not_claim_order_submitted(self):
            code, report, calls, deps = self.run_case(['--mode', 'paper', '--free-tier-confirmed', 'true', '--paper-orders-confirmed', 'true'], signal='HOLD')
            self.assertEqual(report['status'], 'PAPER_NO_NEW_ORDER')
            self.assertFalse(report['execution']['submitted'])
        def test_groq_generation_is_normalized(self):
            class LLM:
                def _generate(self):
                    return 'raw'
            llm = checked_llm(LLM(), NS(normalized_chat_result=lambda result: 'checked-' + result))
            self.assertEqual(llm._generate(), 'checked-raw')
        def test_generation_wrapper_is_not_duplicated(self):
            class LLM:
                def _generate(self):
                    return 'raw'
            q = NS(normalized_chat_result=lambda result: 'checked-' + result)
            llm = LLM(); checked_llm(llm, q); checked_llm(llm, q)
            self.assertEqual(llm._generate(), 'checked-raw')
        def test_free_plan_confirmation_is_required(self):
            code, report, calls, deps = self.run_case(['--mode', 'analyze'])
            self.assertEqual(report['error_code'], 'FREE_PLAN_CONFIRMATION_REQUIRED')
            self.assertEqual(calls, [])
        def macro_case(self, extra_reason=False, receipt_symbol='AAPL'):
            gate = {'passed': False, 'reports': {'macro': 'unverified'},
                    'reasons': ['UNVERIFIED_MACRO'] + (['UNVERIFIED_MARKET'] if extra_reason else []),
                    'accepted_tools': {}}
            payload = {'evidence_quality': gate, 'reports': {'macro_report': 'Source-based macro narrative ' * 5},
                       'model_signal': 'BUY', 'signal': 'NO_TRADE', 'status': 'ANALYSIS_INCOMPLETE'}
            quality = NS(execution_payload=lambda *args: copy.deepcopy(payload), usable_report=lambda value: True,
                TOOLS={'macro': {'get_macro_analysis'}},
                receipt_ok=lambda row, group, symbol: row['status'] == 'success' and row['inputs']['ticker'] == symbol)
            receipt = {'symbol': receipt_symbol, 'tool_name': 'get_macro_analysis', 'status': 'success',
                       'inputs': {'curr_date': '2026-11-12'}, 'output': 'Source observations: rate 3.0, CPI 2.0.'}
            saved = copy.deepcopy(receipt)
            result = prepare(quality, {}, 'BUY', [receipt], 'AAPL')
            self.assertEqual(receipt, saved)
            return result
        def test_global_macro_uses_trusted_attempt_scope(self):
            result = self.macro_case()
            self.assertTrue(result['evidence_quality']['passed'])
            self.assertIn('scope_notes', result['evidence_quality'])
        def test_global_macro_cannot_override_missing_core_evidence(self):
            result = self.macro_case(extra_reason=True)
            self.assertFalse(result['evidence_quality']['passed'])
            self.assertEqual(result['signal'], 'NO_TRADE')
        def test_global_macro_wrong_symbol_not_approved(self):
            result = self.macro_case(receipt_symbol='TSLA')
            self.assertFalse(result['evidence_quality']['passed'])
        def test_failed_graph_does_not_keep_receipts(self):
            with tempfile.TemporaryDirectory() as directory:
                deps, calls = fixture(directory)
                def fail(*args):
                    deps.app.tool_calls_log.append({'status': 'error'})
                    raise ValueError('simulated')
                deps.free_providers._paper_graph_class = object
                deps.free_providers._paper_trade_date = lambda: '2026-11-12'
                deps.free_providers.run_graph = lambda *args: NS(propagate=fail)
                with self.assertRaises(ValueError):
                    run_with_receipts(deps.free_providers, deps.app, 'groq', 'model', 'AAPL')
                self.assertEqual(deps.app.tool_calls_log, [{'old': True}])
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(0 if self_test() else 1)
    raise SystemExit(main())
