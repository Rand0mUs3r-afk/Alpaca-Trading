"""Free-tier Gemini/Groq research with bounded retries; never executes orders."""
import argparse
import copy
import html
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
from types import SimpleNamespace
from urllib.parse import urlparse
from datetime import datetime
from zoneinfo import ZoneInfo

GEMINI_MODELS = ('gemini-3.8-flash', 'gemini-3.6-flash')
GROQ_MODEL = 'openai/gpt-oss-120b'
GROQ_URL = 'https://api.groq.com/openai/v1'
PHASE = 'startup'
ATTEMPTS = []
ACTIVE = {}
LOCK = threading.Lock()
LAST_CALL = 0.0


def http_status(exc):
    seen = set()
    for _ in range(6):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        for value in (getattr(exc, 'code', None), getattr(exc, 'status_code', None),
                      getattr(getattr(exc, 'response', None), 'status_code', None)):
            if not isinstance(value, bool) and str(value).isdigit() and 400 <= int(value) <= 599:
                return int(value)
        exc = exc.__cause__ or exc.__context__
    return None


def can_fallback(exc):
    return http_status(exc) in (404, 408, 429, 500, 502, 503, 504) or isinstance(exc, TimeoutError) or type(exc).__name__ in ('ReadTimeout', 'ConnectTimeout', 'APITimeoutError', 'APIConnectionError')


def bounded_call(call, sleep=time.sleep):
    for attempt in range(2):
        try:
            return call()
        except Exception as exc:
            if attempt or http_status(exc) not in (408, 500, 502, 503, 504):
                raise
            print('AI_RETRY: transient HTTP ' + str(http_status(exc)))
            sleep(20)


def paced(call):
    global LAST_CALL
    with LOCK:
        time.sleep(max(0, 20 - (time.monotonic() - LAST_CALL)))
        LAST_CALL = time.monotonic()
        return bounded_call(call)


def routes(provider, gemini_model, env=None):
    env = os.environ if env is None else env
    if provider not in ('auto', 'gemini', 'groq') or gemini_model not in GEMINI_MODELS:
        raise ValueError('Unsupported provider or model.')
    choices = []
    if provider in ('auto', 'gemini') and env.get('GOOGLE_API_KEY', '').strip():
        choices.append(('google', gemini_model))
    if provider in ('auto', 'groq') and env.get('GROQ_API_KEY', '').strip():
        choices.append(('groq', GROQ_MODEL))
    if not choices:
        raise ValueError('No key is configured for the selected provider.')
    return choices


def choose_analysis(choices, run):
    for index, (provider, model) in enumerate(choices):
        ACTIVE.clear()
        ACTIVE.update(provider=provider, model=model)
        print('AI_PROVIDER: ' + provider + ' / ' + model)
        try:
            result = run(provider, model)
        except Exception as exc:
            ATTEMPTS.append({'provider': provider, 'model': model, 'status': 'failed',
                             'http_status': http_status(exc), 'error_type': type(exc).__name__})
            if index + 1 == len(choices) or not can_fallback(exc):
                raise
            print('AI_FALLBACK: restarting the complete analysis with the next configured provider')
        else:
            ATTEMPTS.append({'provider': provider, 'model': model, 'status': 'completed'})
            return result


def check_url(method, url):
    parsed = urlparse(str(url))
    host = (parsed.hostname or '').lower()
    if host == 'api.groq.com':
        if parsed.scheme != 'https' or parsed.path.rstrip('/') != '/openai/v1/chat/completions' or str(method).upper() != 'POST':
            raise RuntimeError('Only the approved Groq chat endpoint is allowed.')
        return
    if any(host == domain or host.endswith('.' + domain) for domain in
           ('openai.com', 'anthropic.com', 'openrouter.ai', 'api.deepseek.com')):
        raise RuntimeError('Unapproved model provider blocked.')
    if host == 'alpaca.markets' or host.endswith('.alpaca.markets'):
        if host == 'api.alpaca.markets' or str(method).upper() not in ('GET', 'HEAD'):
            raise RuntimeError('Live Alpaca and broker writes are blocked.')


def install_http_guard():
    import requests
    import httpx
    original = requests.sessions.Session.request
    def request(self, method, url, *args, **kwargs):
        check_url(method, url)
        return original(self, method, url, *args, **kwargs)
    requests.sessions.Session.request = request
    original_sync = httpx.Client.send
    def send(self, request, *args, **kwargs):
        check_url(request.method, request.url)
        return original_sync(self, request, *args, **kwargs)
    httpx.Client.send = send
    original_async = httpx.AsyncClient.send
    async def async_send(self, request, *args, **kwargs):
        check_url(request.method, request.url)
        return await original_async(self, request, *args, **kwargs)
    httpx.AsyncClient.send = async_send


def build_llm(provider, model, google_class, groq_class, env=None):
    env = os.environ if env is None else env
    model = model.removeprefix('models/')
    if provider == 'google' and model in GEMINI_MODELS:
        if not env.get('GOOGLE_API_KEY', '').strip():
            raise ValueError('GOOGLE_API_KEY is missing.')
        return google_class(model=model, google_api_key=env['GOOGLE_API_KEY'].strip(),
                            timeout=120, max_retries=1)
    if provider == 'groq' and model == GROQ_MODEL:
        if not env.get('GROQ_API_KEY', '').strip():
            raise ValueError('GROQ_API_KEY is missing.')
        return groq_class(model=model, api_key=env['GROQ_API_KEY'].strip(),
                          base_url=GROQ_URL, timeout=120, max_retries=0,
                          use_responses_api=False, max_tokens=4096, reasoning_effort='low')
    raise ValueError('Model/provider pair is not approved.')


def install_clients():
    from langchain_openai import ChatOpenAI
    from tradingagents.llm_clients.google_client import NormalizedChatGoogleGenerativeAI
    from tradingagents.graph import trading_graph
    from tradingagents.agents.utils import memory
    class PacedGoogle(NormalizedChatGoogleGenerativeAI):
        def _generate(self, *args, **kwargs):
            parent = super()._generate
            return paced(lambda: parent(*args, **kwargs))
    class PacedGroq(ChatOpenAI):
        def _generate(self, *args, **kwargs):
            parent = super()._generate
            return paced(lambda: parent(*args, **kwargs))
    def factory(provider, model, base_url=None, **kwargs):
        if base_url:
            raise ValueError('Custom model endpoints are not permitted.')
        llm = build_llm(provider, model, PacedGoogle, PacedGroq)
        return SimpleNamespace(get_llm=lambda: llm)
    trading_graph.create_llm_client = factory
    trading_graph.FinancialSituationMemory = memory.FinancialSituationMemory
    return trading_graph.TradingAgentsGraph


def run_graph(provider, model, graph_class):
    global PHASE
    from tradingagents.default_config import DEFAULT_CONFIG
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(llm_provider=provider, quick_think_llm=model, deep_think_llm=model,
                  quick_llm_params={}, deep_llm_params={}, backend_url=None,
                  openai_api_key=None, openai_use_local=False, online_tools=True,
                  social_llm_enabled=False,
                  alpaca_use_paper='True', allow_shorts=False,
                  parallel_analysts=False, parallel_risk_first_round=False,
                  tool_semantic_retry_enabled=False,
                  results_dir='eval_results/free-providers/' + provider,
                  agent_memory_dir='.free-actions-memory/' + provider,
                  memory_log_path='.free-actions-memory/' + provider + '/decisions.md')
    PHASE = 'graph_initialization'
    graph = graph_class(config=config)
    PHASE = 'agent_analysis'
    return graph


def error_details(exc):
    status = http_status(exc)
    names = {400: 'AI_INVALID_REQUEST', 401: 'AI_AUTHENTICATION', 403: 'AI_PERMISSION',
             404: 'AI_MODEL_UNAVAILABLE', 408: 'AI_TIMEOUT', 429: 'AI_QUOTA_OR_RATE_LIMIT',
             500: 'AI_SERVER_ERROR', 502: 'AI_SERVER_ERROR', 503: 'AI_SERVER_BUSY', 504: 'AI_TIMEOUT'}
    return {'error_type': type(exc).__name__, 'error_stage': PHASE,
            'http_status': status, 'error_code': names.get(status, 'ERROR_IN_' + PHASE.upper()),
            'error_location': [{'file': Path(frame.filename).name, 'function': frame.name, 'line': frame.lineno}
                               for frame in traceback.extract_tb(exc.__traceback__)[-5:]]}


def main():
    global PHASE
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('check', 'analyze'), default='check')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--model', choices=GEMINI_MODELS, default=GEMINI_MODELS[0])
    parser.add_argument('--provider', choices=('auto', 'gemini', 'groq'), default='auto')
    parser.add_argument('--free-tier-confirmed', choices=('true', 'false'), default='false')
    args = parser.parse_args()
    ATTEMPTS.clear()
    ACTIVE.clear()
    from scripts import free_actions as base
    result = {'status': 'FAILED', 'mode': args.mode, 'symbol': args.ticker,
              'requested_provider': args.provider, 'requested_gemini_model': args.model,
              'orders_enabled': False}
    try:
        PHASE = 'input_validation'
        base.validate(args.ticker, 'check', args.model, False)
        if args.mode == 'analyze':
            if args.free_tier_confirmed != 'true':
                raise ValueError('Confirm supplied AI keys belong to free, unbilled plans.')
            if '@' not in os.getenv('SEC_USER_AGENT', ''):
                raise ValueError('SEC_USER_AGENT requires your contact email.')
            choices = routes(args.provider, args.model)
        PHASE = 'network_guard'
        install_http_guard()
        PHASE = 'paper_account_check'
        result.update(base.paper_check())
        if args.mode == 'analyze':
            from tradingagents.dataflows.config import clear_runtime_api_keys
            clear_runtime_api_keys()
            PHASE = 'free_research_tools'
            base.install_free_tools()
            PHASE = 'local_embeddings'
            base.install_local_memory()
            PHASE = 'client_setup'
            graph_class = install_clients()
            def run(provider, model):
                graph = run_graph(provider, model, graph_class)
                return graph.propagate(args.ticker, datetime.now(ZoneInfo('America/New_York')).date().isoformat())
            state, signal = choose_analysis(choices, run)
            result.update(signal=str(signal), reports={key: str(state.get(key, '')) for key in base.REPORTS})
        result['status'] = 'PAPER_CHECK_OK' if args.mode == 'check' else 'ANALYSIS_COMPLETE_NO_ORDERS'
    except Exception as exc:
        result.update(error_details(exc))
        result['help'] = 'Use http_status and error_code; check free quota or provider access. No paid fallback or orders.'
    result.update(ACTIVE)
    result['provider_attempts'] = ATTEMPTS
    result['limitations'] = ['Research only; no orders.', 'Memory resets each Actions run.',
                            'Free quotas may stop both providers. Provider changes may change decisions.']
    base.REPORT.parent.mkdir(parents=True, exist_ok=True)
    text = base.redact(json.dumps(result, indent=2, ensure_ascii=True))
    base.REPORT.write_text(text + '\n', encoding='utf-8')
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as handle:
            handle.write('# Free AI paper research\n\n<pre>' + html.escape(text) + '</pre>\n')
    print(result['status'])
    return 0 if result['status'] != 'FAILED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
