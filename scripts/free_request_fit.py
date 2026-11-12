"""Fit free Groq requests without deleting evidence; recover wrapped quota errors."""
import copy
import json


def output_limit(payload):
    values = [int(payload.get(key) or 0) for key in ('max_tokens', 'max_completion_tokens')]
    return max(values) or 2048


def adjusted_estimate(payload, estimate):
    # The earlier estimator always reserves at least 2048 output tokens.
    requested = output_limit(payload)
    return estimate(payload) - max(requested, 2048) + requested


def shorten_tool_help(value):
    changed = False
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'description' and isinstance(item, str) and len(item) > 180:
                value[key] = item[:177] + '...'
                changed = True
            elif isinstance(item, (dict, list)):
                changed = shorten_tool_help(item) or changed
    elif isinstance(value, list):
        for item in value:
            changed = shorten_tool_help(item) or changed
    return changed


def fit_payload(payload, limit, estimate, stop):
    result = copy.deepcopy(payload)
    requested = min(output_limit(result), 2048)
    shortened = False
    if adjusted_estimate(result, estimate) > limit:
        shortened = shorten_tool_help(result.get('tools', []))
    input_and_overhead = adjusted_estimate(result, estimate) - output_limit(result)
    available = limit - input_and_overhead - 64
    stats = {'estimated_input_plus_overhead': input_and_overhead,
             'configured_token_budget': limit, 'message_count': len(result.get('messages', [])),
             'tool_count': len(result.get('tools', [])), 'tool_help_shortened': shortened,
             'requested_output_tokens': requested}
    if available < min(512, requested):
        raise stop('GROQ_REQUEST_TOO_LARGE', {
            'quota_source': 'local_preflight', **stats,
            'estimated_request_tokens': adjusted_estimate(result, estimate),
            'quota_help': 'Input is still too large even after shortening tool help. Messages and evidence were not removed.'})
    allowance = min(requested, int(available))
    keys = [key for key in ('max_tokens', 'max_completion_tokens') if key in result] or ['max_tokens']
    for key in keys:
        result[key] = allowance
    stats.update(allowed_output_tokens=allowance, estimated_request_tokens=adjusted_estimate(result, estimate))
    if stats['estimated_request_tokens'] > limit:
        raise stop('GROQ_REQUEST_TOO_LARGE', {'quota_source': 'local_preflight', **stats})
    return result, stats


def recover_error(exc, info, stop_type):
    info = dict(info)
    seen = set()
    for _ in range(10):
        if exc is None or id(exc) in seen:
            break
        seen.add(id(exc))
        if isinstance(exc, stop_type):
            info.update(exc.details)
            info['underlying_error_type'] = 'QuotaStop'
            if str(exc.details.get('quota_source', '')).startswith('local_'):
                info.update(http_status=None, local_status=exc.code, status_is_local=True)
            break
        exc = exc.__cause__ or exc.__context__
    return info


def rebuild_request(request, payload):
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    headers = {key: value for key, value in request.headers.items() if key.lower() != 'content-length'}
    return type(request)(request.method, request.url, headers=headers, content=body,
                         extensions=dict(request.extensions))


def install(free, quota):
    quota.install(free)
    original_send = quota.send_with_limits
    def send(send_function, client, request, args, kwargs, budget, **options):
        payload = json.loads(request.content)
        fitted, stats = fit_payload(payload, budget.limit, quota.estimate_tokens, quota.QuotaStop)
        print('GROQ_REQUEST_SIZE: ' + json.dumps(stats, sort_keys=True))
        if fitted != payload:
            request = rebuild_request(request, fitted)
        options['estimate'] = lambda value: adjusted_estimate(value, quota.estimate_tokens)
        return original_send(send_function, client, request, args, kwargs, budget, **options)
    quota.send_with_limits = send
    original_details = free.error_details
    free.error_details = lambda exc: recover_error(exc, original_details(exc), quota.QuotaStop)


def main():
    from scripts import free_providers, free_groq_limits
    install(free_providers, free_groq_limits)
    return free_providers.main()


if __name__ == '__main__':
    raise SystemExit(main())
