"""Single-provider bounded client; credentials never enter artifacts or errors."""
import json
import time
import requests
from research.common import write_json
from research.claude_context_core import P

MODEL = 'anthropic/claude-sonnet-4.6'
PROVIDER = 'Amazon Bedrock'
TAG = 'amazon-bedrock/global'
BASE = 'https://openrouter.ai/api/v1'
CAP = 1.474196
MAX_CALLS = 3


def payload(system, user):
    schema = dict(type='object', additionalProperties=False,
                  required=['analysis', 'features'], properties={
        'analysis': {'type': 'string'},
        'features': dict(type='array', maxItems=4, items=dict(
            type='object', additionalProperties=False,
            required=['formula', 'rationale'], properties={
                'formula': {'type': 'string'}, 'rationale': {'type': 'string'}}))})
    body = dict(model=MODEL, messages=[dict(role='system', content=system),
                dict(role='user', content=json.dumps(user, ensure_ascii=True))],
                temperature=.2, max_tokens=6000, reasoning={'enabled': False},
                response_format=dict(type='json_schema', json_schema=dict(
                    name='feature_proposals', strict=True, schema=schema)),
                provider=dict(only=[TAG], allow_fallbacks=False,
                    require_parameters=True, data_collection='deny', zdr=True,
                    max_price={'prompt': 3, 'completion': 15}), stream=False)
    size = len(json.dumps(body, ensure_ascii=True).encode())
    assert size <= 80000, 'Payload byte cap exceeded'
    reserve = (size + 2048) * 3e-6 + 6000 * 15e-6
    return body, reserve


def budget_ok(ledger, reserve):
    return len(ledger) < MAX_CALLS and sum(r['reserved_usd'] for r in ledger) + reserve <= CAP


class Client:
    def __init__(self, key):
        self.headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
        self.path = P / 'api_ledger.json'
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else []
        r = requests.get(BASE + '/key', headers=self.headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f'Credential check HTTP {r.status_code}; no inference')
        r = requests.get(BASE + '/endpoints/zdr', timeout=30)
        r.raise_for_status()
        endpoints = [e for e in r.json()['data'] if e.get('model_id') == MODEL
                     and e.get('tag') == TAG and e.get('safety_retention') is False]
        assert len(endpoints) == 1, 'Pinned ZDR endpoint unavailable'
        e = endpoints[0]
        assert float(e['pricing']['prompt']) <= 3e-6
        assert float(e['pricing']['completion']) <= 15e-6
        assert 'response_format' in e['supported_parameters']
        write_json(P / 'provider_preflight.json', dict(checked_unix=time.time(),
                   model=MODEL, endpoint=e, sources=[BASE + '/endpoints/zdr',
                   'https://openrouter.ai/docs/guides/features/zdr'],
                   scope='Aggregate discovery feedback and feature definitions only'))

    def call(self, name, system, user):
        folder = P / 'api'
        folder.mkdir(exist_ok=True)
        output, request = folder / f'{name}_response.json', folder / f'{name}_request.json'
        body, reserve = payload(system, user)
        if output.exists():
            assert json.loads(request.read_text()) == body
            entry = next(e for e in self.ledger if e['name'] == name)
            assert entry['status'] == 'completed'
            return self.content(json.loads(output.read_text()))
        assert not request.exists(), 'Unresolved prior request; no retry'
        assert budget_ok(self.ledger, reserve), 'Budget/call ceiling reached'
        write_json(request, body)
        entry = dict(name=name, reserved_usd=reserve, status='dispatched', started_unix=time.time())
        self.ledger.append(entry)
        write_json(self.path, self.ledger)
        try:
            r = requests.post(BASE + '/chat/completions', headers=self.headers,
                              json=body, timeout=(20, 120))
        except requests.RequestException as exc:
            entry.update(status='network_failure_no_retry', error_type=type(exc).__name__)
            write_json(self.path, self.ledger)
            raise RuntimeError('Network failure; reservation retained, no retry') from None
        entry['http_status'] = r.status_code
        if r.status_code != 200:
            entry['status'] = 'http_failure_no_retry'
            write_json(self.path, self.ledger)
            raise RuntimeError(f'Inference HTTP {r.status_code}; no retry or fallback')
        response = r.json()
        write_json(output, response)
        usage = response.get('usage', {})
        entry.update(status='response_received', provider=response.get('provider'),
                     model=response.get('model'), usage=usage, finished_unix=time.time())
        write_json(self.path, self.ledger)
        assert response.get('provider') == PROVIDER and response.get('model') == MODEL
        assert isinstance(usage.get('cost'), (float, int)) and 0 <= usage['cost'] <= reserve
        answer = self.content(response)
        entry['status'] = 'completed'
        write_json(self.path, self.ledger)
        return answer

    @staticmethod
    def content(response):
        choices = response.get('choices', [])
        assert choices and choices[0].get('finish_reason') == 'stop', 'Incomplete output'
        value = json.loads(choices[0]['message']['content'])
        assert isinstance(value, dict) and isinstance(value.get('analysis'), str)
        fs = value.get('features')
        assert isinstance(fs, list) and len(fs) <= 4
        assert all(isinstance(f, dict) and isinstance(f.get('formula'), str)
                   and isinstance(f.get('rationale'), str) for f in fs)
        return value

    def close(self):
        self.headers.clear()
