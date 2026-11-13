"""Step 1: paper-only long-entry executor, not yet connected to Actions.
Requires independently computed evidence_quality and a structured trade_intent.
Defaults disabled. Dedicated flat paper account only; one symbol/day attempt.
Planned risk is not a guaranteed loss cap. Brackets activate exits after full fill.
Run offline checks with: python -m scripts.paper_executor --self-test
"""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import json
import os
import re
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from zoneinfo import ZoneInfo

D = Decimal
CENT = D('.01')
LOCK = threading.Lock()


class Blocked(ValueError):
    pass


class BrokerError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__('BROKER_HTTP_' + str(status))


def number(value):
    try:
        result = D(str(value))
    except Exception:
        raise Blocked('INVALID_NUMBER') from None
    if not result.is_finite():
        raise Blocked('NONFINITE_NUMBER')
    return result


def stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except Exception:
        raise Blocked('INVALID_TIMESTAMP') from None
    if result.tzinfo is None:
        raise Blocked('TIMESTAMP_NEEDS_TIMEZONE')
    return result


def account_ok(account, expected):
    if not expected or str(account.get('id')) != expected:
        raise Blocked('ACCOUNT_MISMATCH')
    if account.get('status') != 'ACTIVE' or any(account.get(key) for key in
            ('trading_blocked', 'account_blocked', 'trade_suspended_by_user')):
        raise Blocked('ACCOUNT_RESTRICTED')
    if number(account.get('cash', 0)) <= 0 or number(account.get('equity', 0)) <= 0:
        raise Blocked('NO_CASH_OR_EQUITY')


def clock_ok(clock, now):
    if clock.get('is_open') is not True:
        raise Blocked('MARKET_CLOSED')
    age = (now - stamp(clock.get('timestamp'))).total_seconds()
    if not -5 <= age <= 30:
        raise Blocked('STALE_BROKER_CLOCK')
    if (stamp(clock.get('next_close')) - now).total_seconds() < 120:
        raise Blocked('SESSION_ENDING')


def quality_ok(analysis):
    if analysis.get('status') != 'ANALYSIS_COMPLETE_NO_ORDERS':
        raise Blocked('ANALYSIS_NOT_COMPLETE')
    quality = analysis.get('evidence_quality') or {}
    if quality.get('passed') is not True or quality.get('checked_by') != 'deterministic_gate_v1':
        raise Blocked('EVIDENCE_GATE_MISSING_OR_FAILED')
    bad = ('did not return', 'tools were unavailable', 'tools unavailable',
           "i can't provide", 'i cannot provide', "i can't generate", 'technical basis is insufficient')
    for name in ('market', 'news', 'fundamentals'):
        if (quality.get('reports') or {}).get(name) != 'verified':
            raise Blocked('UNVERIFIED_' + name.upper())
        report = str((analysis.get('reports') or {}).get(name + '_report', '')).strip()
        if len(report) < 80 or any(text in report.lower().replace(chr(0x2019), "'") for text in bad):
            raise Blocked('INCOMPLETE_' + name.upper())


def plan(analysis, account, asset, quote, clock, now, expected):
    account_ok(account, expected)
    clock_ok(clock, now)
    quality_ok(analysis)
    intent = analysis.get('trade_intent') or {}
    symbol = str(analysis.get('symbol', ''))
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', symbol):
        raise Blocked('INVALID_SYMBOL')
    if (analysis.get('signal') != 'BUY' or intent.get('action') != 'BUY' or
            intent.get('symbol') != symbol or intent.get('schema_version') != '1.0' or
            intent.get('trading_mode') != 'investment' or intent.get('current_position') != 'NEUTRAL' or
            intent.get('target_position') != 'LONG' or intent.get('position_transition') != 'OPEN_LONG'):
        raise Blocked('INVALID_LONG_ENTRY_INTENT')
    age = (now - stamp(intent.get('generated_at'))).total_seconds()
    if not -5 <= age <= 600:
        raise Blocked('STALE_DECISION')
    if asset.get('symbol') != symbol or asset.get('class') != 'us_equity' or asset.get('status') != 'active' or asset.get('tradable') is not True:
        raise Blocked('ASSET_NOT_ELIGIBLE')
    age = (now - stamp(quote.get('t'))).total_seconds()
    bid, ask = number(quote.get('bp')), number(quote.get('ap'))
    if not -2 <= age <= 15 or bid <= 0 or ask < D('5') or ask < bid:
        raise Blocked('BAD_OR_STALE_QUOTE')
    if (ask - bid) / ((ask + bid) / 2) > D('.003'):
        raise Blocked('SPREAD_TOO_WIDE')
    controls = intent.get('risk_controls') or {}
    stop = number(controls.get('stop_loss_price')).quantize(CENT, rounding=ROUND_DOWN)
    target = number(controls.get('take_profit_price')).quantize(CENT, rounding=ROUND_DOWN)
    entry = (ask * D('1.0015')).quantize(CENT, rounding=ROUND_UP)
    if not (0 < stop < bid <= ask < target and entry < target) or (entry - stop) / entry > D('.06'):
        raise Blocked('INVALID_PROTECTIVE_LEVELS')
    risk = entry - stop * D('.9985')
    reward = target * D('.9985') - entry
    if reward / risk < 2:
        raise Blocked('REWARD_RISK_BELOW_TWO')
    capital = min(number(account['equity']), D('2000'))
    allocation = min(number(account['cash']), capital * D('.25'))
    quantity = int(min(allocation / entry, D('15') / risk).to_integral_value(rounding=ROUND_DOWN))
    if quantity < 1:
        raise Blocked('NO_WHOLE_SHARE_FITS')
    date = stamp(clock['timestamp']).astimezone(ZoneInfo('America/New_York')).date().isoformat()
    client_id = 'paper-v1-' + date + '-' + symbol
    order = dict(symbol=symbol, qty=str(quantity), side='buy', type='limit',
                 limit_price=str(entry), time_in_force='day', order_class='bracket',
                 extended_hours=False, client_order_id=client_id,
                 take_profit={'limit_price': str(target)}, stop_loss={'stop_price': str(stop)})
    return {'order': order, 'planned_risk_usd': str(quantity * risk),
            'notional_usd': str(quantity * entry), 'reward_risk': str(reward / risk)}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise Blocked('BROKER_REDIRECT_BLOCKED')


class PaperBroker:
    """Hardcoded paper REST transport. No live URL, shorting or cancel interface."""
    paper = True
    def __init__(self):
        if os.getenv('ALPACA_USE_PAPER', 'True').strip().lower() != 'true':
            raise Blocked('PAPER_MODE_REQUIRED')
        self.key = os.getenv('ALPACA_API_KEY', '')
        self.secret = os.getenv('ALPACA_SECRET_KEY', '')
        self.expected_id = os.getenv('ALPACA_EXPECTED_PAPER_ACCOUNT_ID', '').strip()
        if not self.key or not self.secret or not self.expected_id:
            raise Blocked('PAPER_CREDENTIALS_MISSING')
        self.opener = build_opener(NoRedirect())

    def _call(self, path, query=None, body=None, data=False):
        base = 'https://data.alpaca.markets' if data else 'https://paper-api.alpaca.markets'
        if not path.startswith('/v2/') or (body is not None and (data or path != '/v2/orders')):
            raise Blocked('ENDPOINT_BLOCKED')
        url = base + path + ('?' + urlencode(query) if query else '')
        request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                          method='POST' if body is not None else 'GET', headers={
                              'APCA-API-KEY-ID': self.key, 'APCA-API-SECRET-KEY': self.secret,
                              'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as exc:
            raise BrokerError(exc.code) from None

    def account(self):
        return self._call('/v2/account')
    def clock(self):
        return self._call('/v2/clock')
    def positions(self):
        return self._call('/v2/positions')
    def open_orders(self):
        return self._call('/v2/orders', {'status': 'open', 'limit': 500})
    def asset(self, symbol):
        return self._call('/v2/assets/' + symbol)
    def quote(self, symbol):
        return self._call('/v2/stocks/quotes/latest', {'symbols': symbol, 'feed': 'iex'}, data=True)['quotes'][symbol]
    def find(self, client_id):
        try:
            return self._call('/v2/orders:by_client_order_id', {'client_order_id': client_id})
        except BrokerError as exc:
            if exc.status == 404:
                return None
            raise
    def submit(self, order):
        return self._call('/v2/orders', body=order)


def execute(analysis, broker, *, enabled=False, now=None):
    """Future caller must compute evidence quality independently, not ask an LLM to approve it.
    File installation alone enables nothing. This first engine supports new long entries
    only; SELL/HOLD do not close positions. Do not share its paper account with another bot.
    """
    if enabled is not True:
        return {'status': 'DISABLED', 'submitted': False}
    if analysis.get('signal') != 'BUY':
        return {'status': 'NO_ACTION', 'submitted': False}
    if not isinstance(broker, PaperBroker) or broker.paper is not True:
        raise Blocked('PAPER_BROKER_REQUIRED')
    supplied_now = now
    now = now or datetime.now(timezone.utc)
    with LOCK:
        account = broker.account()
        account_ok(account, broker.expected_id)
        clock = broker.clock()
        clock_ok(clock, now)
        symbol = str(analysis.get('symbol', ''))
        if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}', symbol):
            raise Blocked('INVALID_SYMBOL')
        date = stamp(clock['timestamp']).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        client_id = 'paper-v1-' + date + '-' + symbol
        existing = broker.find(client_id)
        if existing:
            return {'status': 'EXISTING_ORDER', 'submitted': False, 'client_order_id': client_id}
        if broker.positions() or broker.open_orders():
            raise Blocked('DEDICATED_ACCOUNT_MUST_BE_FLAT')
        asset, quote = broker.asset(symbol), broker.quote(symbol)
        proposal = plan(analysis, account, asset, quote, clock, now, broker.expected_id)
        # Recheck account/clock and other orders immediately before submitting.
        account, clock = broker.account(), broker.clock()
        if broker.positions() or broker.open_orders() or broker.find(client_id):
            raise Blocked('ACCOUNT_CHANGED_BEFORE_SUBMIT')
        final_now = supplied_now or datetime.now(timezone.utc)
        proposal = plan(analysis, account, asset, quote, clock, final_now, broker.expected_id)
        try:
            order = broker.submit(proposal['order'])
        except Exception as exc:
            try:
                existing = broker.find(client_id)
            except Exception:
                existing = None
            if existing:
                return {'status': 'RECONCILED_ORDER_FOUND', 'submitted': True, 'fill_verified': False,
                        'client_order_id': client_id}
            if isinstance(exc, BrokerError) and 400 <= exc.status < 500:
                return {'status': 'REJECTED', 'submitted': False, 'client_order_id': client_id}
            return {'status': 'SUBMISSION_UNKNOWN', 'submitted': None, 'client_order_id': client_id,
                    'help': 'Do not issue a new order ID; reconcile this ID before any retry.'}
        if (not order.get('id') or order.get('client_order_id') != client_id or
                order.get('symbol') != symbol or order.get('side') != 'buy'):
            return {'status': 'SUBMISSION_UNKNOWN', 'submitted': None, 'client_order_id': client_id}
        return {'status': 'ORDER_ACCEPTED_NOT_FILL_VERIFIED', 'submitted': True,
                'fill_verified': False, 'client_order_id': client_id,
                'broker_status': str(order.get('status', 'unknown')), **proposal}


def self_test():
    import copy
    import unittest
    now = stamp('2026-11-12T15:00:00Z')
    def fixture():
        analysis = {'status': 'ANALYSIS_COMPLETE_NO_ORDERS', 'symbol': 'AAPL', 'signal': 'BUY',
            'reports': {key + '_report': 'Verified source observations with numeric values and timestamps. ' * 3
                        for key in ('market', 'news', 'fundamentals')},
            'evidence_quality': {'passed': True, 'checked_by': 'deterministic_gate_v1',
                                 'reports': {'market': 'verified', 'news': 'verified', 'fundamentals': 'verified'}},
            'trade_intent': {'schema_version': '1.0', 'symbol': 'AAPL', 'action': 'BUY',
                'trading_mode': 'investment', 'current_position': 'NEUTRAL', 'target_position': 'LONG',
                'position_transition': 'OPEN_LONG', 'generated_at': now.isoformat(),
                'risk_controls': {'stop_loss_price': 98, 'take_profit_price': 105}}}
        return analysis
    class FakeBroker(PaperBroker):
        def __init__(self):
            self.expected_id = 'paper-test'
            self.sent = []
            self.saved = None
            self.timeout = False
            self.holdings = []
        def account(self):
            return {'id': 'paper-test', 'status': 'ACTIVE', 'cash': '2000', 'equity': '2000'}
        def clock(self):
            return {'is_open': True, 'timestamp': now.isoformat(), 'next_close': '2026-11-12T21:00:00Z'}
        def positions(self):
            return self.holdings
        def open_orders(self):
            return []
        def find(self, client_id):
            return self.saved
        def asset(self, symbol):
            return {'symbol': symbol, 'class': 'us_equity', 'status': 'active', 'tradable': True}
        def quote(self, symbol):
            return {'bp': '99.99', 'ap': '100', 't': now.isoformat()}
        def submit(self, order):
            self.sent.append(copy.deepcopy(order))
            if self.timeout:
                raise TimeoutError()
            self.saved = {'id': 'mock-order', 'client_order_id': order['client_order_id'],
                          'symbol': order['symbol'], 'side': 'buy', 'status': 'accepted'}
            return self.saved
    class Tests(unittest.TestCase):
        def test_disabled(self):
            broker = FakeBroker()
            self.assertEqual(execute(fixture(), broker)['status'], 'DISABLED')
            self.assertFalse(broker.sent)
        def test_hold_does_not_trade(self):
            value = fixture(); value['signal'] = 'HOLD'
            self.assertEqual(execute(value, FakeBroker(), enabled=True, now=now)['status'], 'NO_ACTION')
        def test_complete_buy_builds_protected_cash_order(self):
            broker = FakeBroker()
            result = execute(fixture(), broker, enabled=True, now=now)
            self.assertTrue(result['submitted'])
            self.assertEqual(len(broker.sent), 1)
            self.assertEqual(broker.sent[0]['order_class'], 'bracket')
            self.assertLessEqual(number(result['planned_risk_usd']), 15)
            self.assertLessEqual(number(result['notional_usd']), 500)
        def test_repeat_does_not_submit_twice(self):
            broker = FakeBroker()
            execute(fixture(), broker, enabled=True, now=now)
            self.assertEqual(execute(fixture(), broker, enabled=True, now=now)['status'], 'EXISTING_ORDER')
            self.assertEqual(len(broker.sent), 1)
        def test_missing_quality_blocks(self):
            value = fixture(); value.pop('evidence_quality')
            broker = FakeBroker()
            with self.assertRaises(Blocked): execute(value, broker, enabled=True, now=now)
            self.assertFalse(broker.sent)
        def test_incomplete_report_blocks(self):
            value = fixture(); value['reports']['market_report'] = 'The analyst did not return a complete technical narrative. ' * 3
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_wrong_account_blocks(self):
            broker = FakeBroker(); broker.expected_id = 'wrong'
            with self.assertRaises(Blocked): execute(fixture(), broker, enabled=True, now=now)
        def test_existing_positions_block(self):
            broker = FakeBroker(); broker.holdings = [{'symbol': 'OTHER'}]
            with self.assertRaises(Blocked): execute(fixture(), broker, enabled=True, now=now)
        def test_missing_stop_blocks(self):
            value = fixture(); value['trade_intent']['risk_controls']['stop_loss_price'] = None
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_weak_reward_risk_blocks(self):
            value = fixture(); value['trade_intent']['risk_controls']['take_profit_price'] = 101
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_nonfinite_values_block(self):
            value = fixture(); value['trade_intent']['risk_controls']['stop_loss_price'] = 'NaN'
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_unknown_submit_is_not_retried(self):
            broker = FakeBroker(); broker.timeout = True
            self.assertEqual(execute(fixture(), broker, enabled=True, now=now)['status'], 'SUBMISSION_UNKNOWN')
            self.assertEqual(len(broker.sent), 1)
        def test_closed_market_blocks(self):
            broker = FakeBroker()
            original = broker.clock
            broker.clock = lambda: {**original(), 'is_open': False}
            with self.assertRaises(Blocked): execute(fixture(), broker, enabled=True, now=now)
            self.assertFalse(broker.sent)
        def test_stale_quote_blocks(self):
            broker = FakeBroker()
            broker.quote = lambda symbol: {'bp': 99.99, 'ap': 100, 't': '2026-11-12T14:00:00Z'}
            with self.assertRaises(Blocked): execute(fixture(), broker, enabled=True, now=now)
            self.assertFalse(broker.sent)
        def test_stop_above_current_bid_blocks(self):
            value = fixture(); value['trade_intent']['risk_controls']['stop_loss_price'] = 100.10
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_typographic_refusal_blocks(self):
            value = fixture()
            value['reports']['fundamentals_report'] = ('I can' + chr(0x2019) + 't provide that. ') * 8
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
        def test_stale_analysis_blocks(self):
            value = fixture(); value['trade_intent']['generated_at'] = '2026-11-11T15:00:00Z'
            with self.assertRaises(Blocked): execute(value, FakeBroker(), enabled=True, now=now)
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful()


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('Not connected to Actions. Only --self-test is supported as a command.')
    raise SystemExit(0 if self_test() else 1)
