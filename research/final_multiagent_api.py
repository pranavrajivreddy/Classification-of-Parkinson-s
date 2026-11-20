"""Bounded OpenRouter client. No credentials in files, prompts or diagnostics."""
import json,time
import requests
from research.common import write_json
from research.final_multiagent_core import P

MODEL='deepseek/deepseek-v3.2'
BASE='https://openrouter.ai/api/v1'
CAP=.50
def payload(system,user):
 out=dict(model=MODEL,messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(user,ensure_ascii=True))],temperature=.2,max_tokens=4096,reasoning={'enabled':False},response_format={'type':'json_object'},provider={'only':['deepinfra'],'allow_fallbacks':False,'require_parameters':True,'data_collection':'deny','zdr':True,'max_price':{'prompt':1,'completion':1}},stream=False)
 size=len(json.dumps(out,ensure_ascii=True).encode());assert size<=40000,'Request too large'
 return out,(size+4096+1024)/1e6

class Client:
 def __init__(self,key):
  self._headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
  self.path=P/'api_ledger.json';self.ledger=json.loads(self.path.read_text()) if self.path.exists() else []
  r=requests.get(BASE+'/key',headers=self._headers,timeout=30)
  if r.status_code!=200:raise RuntimeError(f'OpenRouter credential check failed HTTP {r.status_code}; no inference sent')
  # Never persist response fields that might identify the key or account.
  r=requests.get(BASE+'/models/'+MODEL+'/endpoints',timeout=30);r.raise_for_status();meta=r.json()
  endpoints=[e for e in meta['data']['endpoints'] if e['provider_name']=='DeepInfra']
  assert endpoints and any(float(e['pricing']['prompt'])<=1e-6 and float(e['pricing']['completion'])<=1e-6 and 'response_format' in e['supported_parameters'] for e in endpoints),'Requested provider/price/JSON unavailable'
  write_json(P/'provider_preflight.json',dict(checked_unix=time.time(),model=MODEL,requested_provider='deepinfra',eligible_endpoints=endpoints,privacy_sources=['https://openrouter.ai/docs/guides/privacy/data-collection','https://openrouter.ai/docs/guides/features/zdr','https://deepinfra.com/privacy'],scope='Feature definitions/formulas and aggregate discovery metrics only; no participant records'))
 def call(self,name,system,user):
  output=P/'api'/f'{name}_response.json';output.parent.mkdir(parents=True,exist_ok=True)
  if output.exists():return self.content(json.loads(output.read_text()))
  assert len(self.ledger)<8,'Maximum request attempts reached'
  body,reserve=payload(system,user)
  assert sum(r['reserved_usd'] for r in self.ledger)+reserve<=CAP,'Spending reservation cap reached'
  request=P/'api'/f'{name}_request.json';assert not request.exists(),'Prior request unresolved: no automatic retry'
  write_json(request,body);entry=dict(name=name,reserved_usd=reserve,status='dispatched',started_unix=time.time());self.ledger.append(entry);write_json(self.path,self.ledger)
  try:r=requests.post(BASE+'/chat/completions',headers=self._headers,json=body,timeout=(20,120))
  except requests.RequestException as exc:
   entry['status']='network_failure_no_retry';entry['error_type']=type(exc).__name__;write_json(self.path,self.ledger);raise RuntimeError('API network failure; reservation retained, no retry') from None
  entry['http_status']=r.status_code
  if r.status_code!=200:
   entry['status']='http_failure_no_retry';write_json(self.path,self.ledger);raise RuntimeError(f'API failed HTTP {r.status_code}; no retry or provider fallback')
  response=r.json();write_json(output,response);usage=response.get('usage',{});entry.update(status='completed',provider=response.get('provider'),model=response.get('model'),usage=usage,finished_unix=time.time());write_json(self.path,self.ledger)
  assert response.get('provider')=='DeepInfra','Unexpected provider; stop'
  return self.content(response)
 @staticmethod
 def content(response):
  choices=response.get('choices',[]);assert choices and choices[0].get('finish_reason')=='stop','Incomplete API output; stop'
  text=choices[0]['message']['content'];assert isinstance(text,str)
  return json.loads(text)
 def close(self):self._headers.clear()
