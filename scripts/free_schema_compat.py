"""Groq wire-schema compatibility; retain original Pydantic validation."""
import copy
import json

SCALAR_KEYS = {'type', 'enum', 'format', 'pattern', 'minLength', 'maxLength',
               'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
               'multipleOf', 'title', 'description', 'default', 'examples'}
ANNOTATIONS = {'title', 'description', 'default', 'examples'}


def resolve_local(branch, root):
    result = branch
    seen = set()
    while isinstance(result, dict) and '$ref' in result:
        ref = result['$ref']
        if not isinstance(ref, str) or not ref.startswith('#/') or ref in seen:
            return None
        if set(result) - ({'$ref'} | ANNOTATIONS):
            return None
        seen.add(ref)
        result = root
        for part in ref[2:].split('/'):
            if not isinstance(result, dict):
                return None
            result = result.get(part.replace('~1', '/').replace('~0', '~'))
    return result


def normalize_nullable_scalars(schema):
    root = copy.deepcopy(schema)
    changes = 0
    def visit(node):
        nonlocal changes
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        branches = node.get('anyOf')
        if isinstance(branches, list) and len(branches) == 2 and set(node) <= ({'anyOf'} | ANNOTATIONS):
            resolved = [resolve_local(branch, root) for branch in branches]
            nulls = [item for item in resolved if isinstance(item, dict) and item.get('type') == 'null' and set(item) <= ({'type'} | ANNOTATIONS)]
            values = [item for item in resolved if isinstance(item, dict) and item.get('type') in ('string', 'number', 'integer', 'boolean') and set(item) <= SCALAR_KEYS]
            if len(nulls) == 1 and len(values) == 1:
                value = copy.deepcopy(values[0])
                enum = value.get('enum')
                if enum is None or (isinstance(enum, list) and all(isinstance(item, (str, int, float, bool)) or item is None for item in enum)):
                    annotations = {key: item for key, item in node.items() if key != 'anyOf'}
                    value['type'] = [value['type'], 'null']
                    if enum is not None:
                        value['enum'] = enum if None in enum else enum + [None]
                    value.update(annotations)
                    node.clear()
                    node.update(value)
                    changes += 1
        for child in list(node.values()):
            visit(child)
    visit(root)
    return root, changes


def normalize_payload(payload):
    result = copy.deepcopy(payload)
    response = result.get('response_format', {})
    if response.get('type') != 'json_schema':
        return result, 0
    spec = response.get('json_schema', {})
    if not isinstance(spec.get('schema'), dict):
        return result, 0
    spec['schema'], count = normalize_nullable_scalars(spec['schema'])
    return result, count


def compact_manager_history(text):
    if any('{' + name + '}' in text for name in ('risk_debate_digest', 'investment_debate_digest', 'research_debate_digest')):
        for name in ('history', 'bull_history', 'bear_history', 'risky_history', 'safe_history', 'neutral_history'):
            text = text.replace('{' + name + '}', '[Use the recent manager debate digest; full history is not repeated.]')
    return text


def install(free, compact, fit, quota):
    original_template = compact.compact_template
    compact.compact_template = lambda text: compact_manager_history(original_template(text))
    compact.install(free, fit, quota)
    original_send = quota.send_with_limits
    def send(send_function, client, request, args, kwargs, budget, **options):
        payload, count = normalize_payload(json.loads(request.content))
        if count:
            print('GROQ_NULLABLE_SCHEMA_FIELDS: ' + str(count))
            request = fit.rebuild_request(request, payload)
        return original_send(send_function, client, request, args, kwargs, budget, **options)
    quota.send_with_limits = send


def main():
    from scripts import free_providers, free_compact_context, free_request_fit, free_groq_limits
    install(free_providers, free_compact_context, free_request_fit, free_groq_limits)
    return free_providers.main()


if __name__ == '__main__':
    raise SystemExit(main())
