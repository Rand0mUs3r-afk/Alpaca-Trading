"""Add secret-safe stage diagnostics and equivalent Gemini model-name aliases."""
import functools
import json
from pathlib import Path
import traceback

STAGE = 'startup'
FAILURE = None


def model_aliases(names):
    canonical = tuple(name for name in names if not name.startswith('models/'))
    return tuple(dict.fromkeys(canonical + tuple('models/' + name for name in canonical)))


def capture_failure(exc):
    global FAILURE
    codes = {
        "Model is not on this runner's allowlist.": 'MODEL_NAME_FORMAT',
        'Confirm your Google API project has billing disabled.': 'FREE_TIER_NOT_CONFIRMED',
        'SEC_USER_AGENT must contain your contact email.': 'SEC_CONTACT_MISSING',
        'Missing a required Actions secret. See FREE_ACTIONS.md.': 'REQUIRED_SECRET_MISSING',
        'The paper account does not match your expected account ID.': 'PAPER_ACCOUNT_MISMATCH',
    }
    FAILURE = {'error_stage': STAGE,
               'error_code': codes.get(str(exc), 'ERROR_IN_' + STAGE.upper()),
               'error_location': [
                   {'file': Path(frame.filename).name, 'function': frame.name, 'line': frame.lineno}
                   for frame in traceback.extract_tb(exc.__traceback__)[-5:]]}
    errors = getattr(exc, 'errors', None)
    if callable(errors):
        try:
            rows = errors(include_input=False, include_url=False)
            FAILURE['validation_error_types'] = sorted({
                str(row.get('type', 'unknown'))[:100] for row in rows[:10]})
        except Exception:
            pass
    # Never publish exception messages, input values, headers or response bodies.


def observed(stage, function=None):
    if function is None:
        return functools.partial(observed, stage)
    @functools.wraps(function)
    def call(*args, **kwargs):
        global STAGE
        STAGE = stage
        print('STAGE: ' + stage)
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            capture_failure(exc)
            raise
    return call


def enrich_result(text):
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        return text
    result['last_stage'] = STAGE
    if result.get('status') == 'FAILED':
        result.update(FAILURE or {'error_stage': STAGE, 'error_code': 'UNCLASSIFIED_FAILURE'})
        result['help'] = 'Report error_stage, error_code and error_location. Do not share keys. No orders were enabled.'
    return json.dumps(result, indent=2, ensure_ascii=True)


def install(base):
    global STAGE, FAILURE
    STAGE, FAILURE = 'startup', None
    # Same two models, with or without Google's optional resource-name prefix.
    base.MODELS = model_aliases(base.MODELS)
    for name, stage in (
        ('validate', 'input_validation'), ('install_http_guard', 'network_guard'),
        ('paper_check', 'paper_account_check'), ('analyze', 'analysis_imports'),
        ('install_free_tools', 'free_research_tools'),
    ):
        setattr(base, name, observed(stage, getattr(base, name)))
    memory_install = base.install_local_memory
    @observed('local_embeddings')
    def memory_and_graph():
        global STAGE
        memory_install()
        STAGE = 'graph_import'
        print('STAGE: graph_import')
        from tradingagents.graph import trading_graph
        from tradingagents.agents.utils import memory
        # Ensure the graph uses the local memory class even if imported earlier.
        trading_graph.FinancialSituationMemory = memory.FinancialSituationMemory
        graph = trading_graph.TradingAgentsGraph
        graph.__init__ = observed('graph_initialization', graph.__init__)
        graph.propagate = observed('agent_analysis', graph.propagate)
    base.install_local_memory = memory_and_graph
    redact = base.redact
    base.redact = lambda text: redact(enrich_result(text))


def main():
    from scripts import free_actions
    install(free_actions)
    return free_actions.main()


if __name__ == '__main__':
    raise SystemExit(main())
