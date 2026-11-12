"""Groq-only source prompt profile: selected evidence instead of repeated full reports."""
import importlib
import os
from pathlib import Path
import tempfile

PROFILE = {
    'report_context_budget_tokens': 1600,
    'report_context_max_chunks': 6,
    'report_context_min_chunks_per_report': 1,
    'report_context_max_points_per_report': 4,
    'report_context_max_claims_per_report': 4,
    'report_context_point_chars': 180,
    'report_context_claim_chars': 180,
    'report_context_compact_points_per_report': 2,
    'report_context_compact_point_chars': 140,
    'report_context_compact_excerpt_chars': 180,
    'report_context_compact_max_excerpts': 6,
    'report_context_scoreboard_claims_per_side': 2,
    'report_context_memory_chars': 4000,
    'debate_digest_max_messages': 4,
    'debate_digest_message_chars': 360,
    'debate_digest_total_chars': 1800,
}


def compact_template(text):
    # Do not remove full reports unless a claim matrix is available in this template.
    if '{claim_matrix}' in text and '{all_reports_text}' in text:
        text = text.replace('{all_reports_text}',
            '[Full reports are retained outside this request. Use the selected claim matrix; '
            'it may omit evidence. Prefer HOLD/NEUTRAL if evidence is insufficient.]')
    # Keep the digest and latest counterarguments, rather than duplicating full history.
    if '{debate_digest}' in text:
        for field in ('history', 'bull_history', 'bear_history', 'risky_history', 'safe_history', 'neutral_history'):
            text = text.replace('{' + field + '}', '[Use the recent debate digest instead of repeated full history.]')
    return text


def write_profile(loader, destination):
    changed = 0
    for name in loader.list_prompt_templates():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe prompt-template path.')
        original = loader.load_prompt(name)
        compact = compact_template(original)
        changed += compact != original
        path = Path(destination) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(compact, encoding='utf-8')
    if not changed:
        raise RuntimeError('No compatible report/history templates found; profile not applied.')
    return changed


class ProfileGraph:
    def __init__(self, graph):
        self.graph = graph
    def __getattr__(self, name):
        return getattr(self.graph, name)
    def propagate(self, *args, **kwargs):
        loader = importlib.import_module('tradingagents.prompts.loader')
        context = importlib.import_module('tradingagents.agents.utils.report_context')
        previous_dir = os.environ.get('TRADINGAGENTS_PROMPT_DIR')
        previous_defaults = dict(context.DEFAULT_CONTEXT_CONFIG)
        with tempfile.TemporaryDirectory(prefix='groq-prompts-') as directory:
            count = write_profile(loader, directory)
            try:
                os.environ['TRADINGAGENTS_PROMPT_DIR'] = directory
                # Researchers that omit config must receive the same compact defaults.
                context.DEFAULT_CONTEXT_CONFIG.update(PROFILE)
                print('GROQ_COMPACT_PROMPT_TEMPLATES: ' + str(count))
                return self.graph.propagate(*args, **kwargs)
            finally:
                context.DEFAULT_CONTEXT_CONFIG.clear()
                context.DEFAULT_CONTEXT_CONFIG.update(previous_defaults)
                if previous_dir is None:
                    os.environ.pop('TRADINGAGENTS_PROMPT_DIR', None)
                else:
                    os.environ['TRADINGAGENTS_PROMPT_DIR'] = previous_dir


def install(free, fit, quota):
    fit.install(free, quota)
    original = free.run_graph
    def run(provider, model, graph_class):
        if provider != 'groq':
            return original(provider, model, graph_class)
        def build(config):
            config = dict(config)
            config.update(PROFILE)
            return graph_class(config=config)
        return ProfileGraph(original(provider, model, build))
    free.run_graph = run


def main():
    from scripts import free_providers, free_request_fit, free_groq_limits
    install(free_providers, free_request_fit, free_groq_limits)
    return free_providers.main()


if __name__ == '__main__':
    raise SystemExit(main())
