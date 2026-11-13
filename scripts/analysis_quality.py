"""Tool-call compatibility and conservative report/provenance checks.
'verified' means an acceptable report plus a matching successful tool receipt,
not proof that every natural-language claim is correct or a trade will profit.
Receipts must come from the application's tool runner, never model assertions.
Run offline checks: python -m scripts.analysis_quality --self-test
"""
import copy
import json
import math
import re

REQUIRED = ('market', 'news', 'fundamentals')
TOOLS = {
    'market': {'get_technical_brief', 'get_alpaca_data_report', 'get_alpaca_data',
               'get_stockstats_indicators_report_online'},
    'news': {'get_google_news', 'get_finnhub_news_recent', 'get_finnhub_news', 'get_stock_news_openai'},
    'fundamentals': {'get_fundamentals_openai', 'get_simfin_balance_sheet',
                     'get_simfin_cashflow', 'get_simfin_income_stmt'},
    'sentiment': {'get_reddit_stock_info', 'get_reddit_news'},
    'macro': {'get_macro_analysis', 'get_economic_indicators', 'get_yield_curve_analysis',
              'get_macro_news_openai', 'get_global_macro_indicators'},
}
BAD = ('did not return', 'tools were unavailable', 'tools unavailable',
       "i can't provide", 'i cannot provide', "i can't generate", 'please provide the analysis',
       'technical basis is insufficient', 'tool-loop halted', 'analysis was limited',
       'analysis encountered an error and could not complete successfully',
       'unable to perform fundamental analysis', 'incomplete analysis')
FAILED_REPORT_HEADING = re.compile(
    r'^\s*#{1,6}\s*(?:fundamentals?|market|technical|news|social|sentiment|macro)\s+(?:analysis\s+)?error\b',
    re.IGNORECASE | re.MULTILINE)


class InvalidModelReply(ValueError):
    def __init__(self, code):
        self.error_code = code
        super().__init__(code)


def visible_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return '\n'.join(block.get('text', '') for block in content
                         if isinstance(block, dict) and block.get('type') in ('text', 'output_text')
                         and isinstance(block.get('text'), str)).strip()
    return ''


def normalized_message(message, generation_info=None):
    metadata = dict(getattr(message, 'response_metadata', {}) or {})
    metadata.update(generation_info or {})
    finish = str(metadata.get('finish_reason') or metadata.get('stop_reason') or '').lower()
    if finish in ('length', 'max_tokens', 'max_output_tokens'):
        raise InvalidModelReply('MODEL_RESPONSE_TRUNCATED')
    extra = dict(getattr(message, 'additional_kwargs', {}) or {})
    if extra.get('refusal') or finish in ('content_filter', 'safety'):
        raise InvalidModelReply('MODEL_REFUSED')
    if getattr(message, 'invalid_tool_calls', None):
        raise InvalidModelReply('INVALID_TOOL_CALL')
    canonical = list(getattr(message, 'tool_calls', []) or [])
    if canonical:
        raw = []
        for call in canonical:
            if not isinstance(call, dict):
                raise InvalidModelReply('INVALID_TOOL_CALL')
            name, call_id, args = call.get('name'), call.get('id'), call.get('args', {})
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,127}', name):
                raise InvalidModelReply('INVALID_TOOL_NAME')
            if not isinstance(call_id, str) or not call_id or len(call_id) > 128:
                raise InvalidModelReply('INVALID_TOOL_ID')
            if not isinstance(args, dict):
                raise InvalidModelReply('INVALID_TOOL_ARGUMENTS')
            raw.append({'id': call_id, 'type': 'function',
                        'function': {'name': name, 'arguments': json.dumps(args, allow_nan=False)}})
        # Older analysts look here, while modern LangChain also exposes .tool_calls.
        extra['tool_calls'] = raw
    content = visible_text(getattr(message, 'content', ''))
    if not content and not extra.get('tool_calls'):
        raise InvalidModelReply('EMPTY_MODEL_RESPONSE')
    if hasattr(message, 'model_copy'):
        return message.model_copy(update={'content': content, 'additional_kwargs': extra})
    result = copy.copy(message)
    result.content, result.additional_kwargs = content, extra
    return result


def normalized_chat_result(result):
    generations = list(getattr(result, 'generations', []) or [])
    if not generations:
        raise InvalidModelReply('EMPTY_MODEL_RESULT')
    converted = []
    for generation in generations:
        message = normalized_message(generation.message, getattr(generation, 'generation_info', None))
        if hasattr(generation, 'model_copy'):
            converted.append(generation.model_copy(update={'message': message}))
        else:
            new = copy.copy(generation)
            new.message = message
            converted.append(new)
    if hasattr(result, 'model_copy'):
        return result.model_copy(update={'generations': converted})
    new = copy.copy(result)
    new.generations = converted
    return new


def clean_text(value):
    return str(value or '').replace(chr(0x2019), "'").strip()


def usable_report(value):
    text = clean_text(value)
    return (len(text) >= 80 and not FAILED_REPORT_HEADING.search(text)
            and not any(marker in text.lower() for marker in BAD))


def receipt_ok(receipt, group, symbol):
    if receipt.get('tool_name') not in TOOLS[group] or receipt.get('status') != 'success':
        return False
    if receipt.get('symbol') not in (None, symbol):
        return False
    inputs = receipt.get('inputs') or {}
    supplied = inputs.get('ticker') or inputs.get('symbol')
    query = str(inputs.get('query', ''))
    if supplied:
        if str(supplied).upper() != symbol:
            return False
    elif symbol not in re.findall(r'[A-Z][A-Z0-9.\-]*', query.upper()):
        return False
    if (receipt.get('quality_details') or {}).get('is_suspect') or receipt.get('error_details'):
        return False
    output = receipt.get('output')
    text = clean_text(output)
    if len(text) < 40 or any(word in text.lower() for word in
           ('error:', 'timeout', 'unavailable', 'file missing', 'no report available',
            'no data available', 'no records found', 'no news items found', 'api key not found')):
        return False
    try:
        data = output if isinstance(output, dict) else json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        if data.get('available') is False:
            return False
        if group == 'fundamentals' and 'facts' in data:
            rows = [row for values in data['facts'].values() if isinstance(values, list) for row in values]
            return any(isinstance(row, dict) and not isinstance(row.get('val'), bool)
                       and isinstance(row.get('val'), (int, float)) and math.isfinite(row['val']) for row in rows)
        if group == 'news' and 'items' in data:
            return any(isinstance(row, dict) and row.get('title') and str(row.get('url', '')).startswith(('https://', 'http://'))
                       for row in data['items'])
    return True


def assess(reports, receipts, symbol):
    """Receipts must be cleared/scoped by the caller for each symbol/provider attempt."""
    symbol = str(symbol).upper()
    statuses, accepted, reasons = {}, {}, []
    for group in TOOLS:
        report = clean_text(reports.get(group + '_report'))
        matches = [receipt for receipt in receipts if receipt_ok(receipt, group, symbol)]
        accepted[group] = sorted({receipt['tool_name'] for receipt in matches})
        if usable_report(report) and matches:
            statuses[group] = 'verified'
        elif group not in REQUIRED and (not report or report.upper().startswith('UNAVAILABLE:')):
            statuses[group] = 'unavailable'
        else:
            statuses[group] = 'unverified'
            reasons.append('UNVERIFIED_' + group.upper())
    return {'checked_by': 'deterministic_gate_v1', 'passed': not reasons,
            'reports': statuses, 'accepted_tools': accepted, 'reasons': reasons,
            'meaning': 'Report completeness and tool provenance only; not verification of every claim.'}


def execution_payload(state, signal, receipts, symbol):
    reports = {group + '_report': str(state.get(group + '_report', '')) for group in TOOLS}
    reports['final_trade_decision'] = str(state.get('final_trade_decision', ''))
    quality = assess(reports, receipts, symbol)
    intent = state.get('final_trade_intent')
    if hasattr(intent, 'model_dump'):
        intent = intent.model_dump(mode='json')
    intent = copy.deepcopy(intent) if isinstance(intent, dict) else None
    action = str(signal).upper()
    if action not in ('BUY', 'HOLD', 'SELL', 'LONG', 'SHORT', 'NEUTRAL'):
        quality['passed'] = False
        quality['reasons'].append('INVALID_MODEL_SIGNAL')
    if action == 'BUY' and not intent:
        quality['passed'] = False
        quality['reasons'].append('STRUCTURED_INTENT_MISSING')
    return {'status': 'ANALYSIS_COMPLETE_NO_ORDERS' if quality['passed'] else 'ANALYSIS_INCOMPLETE',
            'symbol': symbol, 'signal': action if quality['passed'] else 'NO_TRADE',
            'model_signal': action, 'reports': reports, 'evidence_quality': quality,
            'trade_intent': intent, 'orders_enabled': False}


def self_test():
    from types import SimpleNamespace as NS
    import unittest
    def message(**kwargs):
        fields = dict(content='', tool_calls=[], invalid_tool_calls=[], additional_kwargs={}, response_metadata={})
        fields.update(kwargs)
        return NS(**fields)
    def fixtures():
        reports = {name + '_report': 'Source-based narrative with actual observations and explicit limitations noted for review. ' * 2
                   for name in REQUIRED}
        receipts = []
        for name, tool in [('market', 'get_technical_brief'), ('news', 'get_stock_news_openai'), ('fundamentals', 'get_fundamentals_openai')]:
            output = {'available': True, 'facts': {'Revenue:USD': [{'val': 123456}]}} if name == 'fundamentals' else (
                {'available': True, 'items': [{'title': 'Actual item', 'url': 'https://example.com/source'}]} if name == 'news' else
                {'price': 100, 'rsi': 55, 'source': 'test bars', 'timestamp': '2026-11-12'})
            receipts.append({'symbol': 'AAPL', 'tool_name': tool, 'status': 'success',
                             'inputs': {'ticker': 'AAPL'}, 'output': json.dumps(output)})
        return reports, receipts
    class Tests(unittest.TestCase):
        def test_canonical_tool_call_gets_legacy_wire_form(self):
            original = message(); original.tool_calls = [{'id': 'call_1', 'name': 'get_technical_brief', 'args': {'ticker': 'AAPL'}}]
            result = normalized_message(original)
            raw = result.additional_kwargs['tool_calls'][0]
            self.assertEqual(raw['id'], 'call_1')
            self.assertEqual(json.loads(raw['function']['arguments']), {'ticker': 'AAPL'})
            self.assertEqual(original.additional_kwargs, {})
        def test_empty_plain_reply_is_rejected(self):
            with self.assertRaises(InvalidModelReply): normalized_message(message())
        def test_truncated_reply_is_rejected(self):
            value = message(); value.content = 'Partial analysis'
            with self.assertRaises(InvalidModelReply): normalized_message(value, {'finish_reason': 'length'})
        def test_invalid_tool_call_is_rejected(self):
            value = message(); value.invalid_tool_calls = [{'name': 'broken'}]
            with self.assertRaises(InvalidModelReply): normalized_message(value)
        def test_missing_tool_id_is_rejected(self):
            value = message(); value.tool_calls = [{'name': 'get_data', 'args': {}}]
            with self.assertRaises(InvalidModelReply): normalized_message(value)
        def test_visible_blocks_are_normalized(self):
            value = message(); value.content = [{'type': 'text', 'text': 'Actual narrative'}]
            self.assertEqual(normalized_message(value).content, 'Actual narrative')
        def test_chat_result_retains_type_and_metadata(self):
            value = message(); value.content = 'Actual narrative'
            original = NS(generations=[NS(message=value, generation_info={'finish_reason': 'stop'})], llm_output={'tokens': 5})
            result = normalized_chat_result(original)
            self.assertEqual(result.llm_output, original.llm_output)
            self.assertIsNot(result.generations[0], original.generations[0])
        def test_reports_without_receipts_fail(self):
            reports, receipts = fixtures()
            self.assertFalse(assess(reports, [], 'AAPL')['passed'])
        def test_required_reports_and_matching_receipts_pass(self):
            reports, receipts = fixtures()
            self.assertTrue(assess(reports, receipts, 'AAPL')['passed'])
        def test_incomplete_technical_report_fails(self):
            reports, receipts = fixtures(); reports['market_report'] = 'The analyst did not return a complete technical narrative. ' * 3
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_failed_receipt_is_not_success(self):
            reports, receipts = fixtures(); receipts[0]['status'] = 'error'
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_wrong_symbol_receipts_fail(self):
            reports, receipts = fixtures(); receipts[0]['inputs']['ticker'] = 'TSLA'
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_empty_sec_facts_fail(self):
            reports, receipts = fixtures(); receipts[2]['output'] = json.dumps({'available': True, 'facts': {'Revenue:USD': []}})
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_missing_news_items_fail(self):
            reports, receipts = fixtures(); receipts[1]['output'] = json.dumps({'available': False, 'items': []})
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_optional_guess_without_provenance_fails(self):
            reports, receipts = fixtures(); reports['sentiment_report'] = 'Social bullish mentions supposedly 55 percent. ' * 4
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_explicit_optional_unavailable_is_allowed(self):
            reports, receipts = fixtures(); reports['sentiment_report'] = 'UNAVAILABLE: No verified social data.'
            self.assertTrue(assess(reports, receipts, 'AAPL')['passed'])
        def test_buy_without_structured_intent_is_not_executable(self):
            reports, receipts = fixtures()
            result = execution_payload(reports, 'BUY', receipts, 'AAPL')
            self.assertEqual(result['status'], 'ANALYSIS_INCOMPLETE')
            self.assertEqual(result['signal'], 'NO_TRADE')
        def test_refusal_metadata_is_rejected(self):
            with self.assertRaises(InvalidModelReply) as caught:
                normalized_message(message(additional_kwargs={'refusal': 'cannot comply'}))
            self.assertEqual(caught.exception.error_code, 'MODEL_REFUSED')
        def test_unavailable_statement_receipt_fails(self):
            reports, receipts = fixtures()
            receipts[2]['tool_name'] = 'get_simfin_balance_sheet'
            receipts[2]['output'] = 'Annual balance sheet unavailable because the SimFin file is missing. ' * 2
            self.assertFalse(assess(reports, receipts, 'AAPL')['passed'])
        def test_never_extracts_trade_intent_from_prose(self):
            reports, receipts = fixtures(); reports['final_trade_decision'] = 'BUY with stop 98 and target 105'
            self.assertIsNone(execution_payload(reports, 'BUY', receipts, 'AAPL')['trade_intent'])
        def test_legacy_fundamentals_failure_with_successful_receipt_is_blocked(self):
            reports, receipts = fixtures()
            reports['fundamentals_report'] = ('# Fundamentals Analysis Error\n\n'
                'The fundamentals analysis encountered an error and could not complete successfully.\n'
                'Unable to perform fundamental analysis for this symbol. Incomplete Analysis.')
            result = assess(reports, receipts, 'AAPL')
            self.assertFalse(result['passed'])
            self.assertEqual(result['reports']['fundamentals'], 'unverified')
            self.assertEqual(result['accepted_tools']['fundamentals'], ['get_fundamentals_openai'])
        def test_failure_headings_block_other_reports(self):
            for name in ('Market', 'News', 'Social', 'Macro'):
                with self.subTest(name=name):
                    self.assertFalse(usable_report('# ' + name + ' Analysis Error\n' + 'Failure details for review. ' * 5))
        def test_explicit_incomplete_analysis_blocks(self):
            self.assertFalse(usable_report('Incomplete Analysis. ' + 'There is no completed analyst narrative. ' * 5))
        def test_ordinary_discussion_of_source_error_is_not_automatically_blocked(self):
            self.assertTrue(usable_report('A source error was resolved by a successful retry. '
                'The completed report below uses the resulting financial observations. ' * 2))
        def test_failed_fundamentals_buy_returns_no_trade(self):
            reports, receipts = fixtures()
            reports['fundamentals_report'] = '# Fundamentals Analysis Error\n' + 'Provider failure prevented completion. ' * 5
            reports['final_trade_intent'] = {'action': 'BUY', 'symbol': 'AAPL'}
            result = execution_payload(reports, 'BUY', receipts, 'AAPL')
            self.assertEqual(result['status'], 'ANALYSIS_INCOMPLETE')
            self.assertEqual(result['signal'], 'NO_TRADE')
            self.assertEqual(result['model_signal'], 'BUY')
            self.assertFalse(result['orders_enabled'])
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('Helper only; only --self-test is available as a command.')
    raise SystemExit(0 if self_test() else 1)
