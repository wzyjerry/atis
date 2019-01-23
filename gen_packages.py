import os
import pickle

DATA_DIR = 'data'

def load_dicts(fname='atis.train.pkl'):
  with open(fname, 'rb') as stream:
    _, dicts = pickle.load(stream)
  print('Done  loading: ', fname)
  return dicts

dicts = load_dicts(os.path.join(DATA_DIR, 'atis.train.pkl'))

t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
i2t, i2s, i2in = map(lambda d: {d[k]: k for k in d.keys()}, [t2i, s2i, in2i])

entitylist = [{
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4832c5c4952fdf7150e1af",
	"name": "aircraft_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4832fcc4952fdf7150e1b3",
	"name": "airline_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483308c4952fdf7150e1b4",
	"name": "airline_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483315c4952fdf7150e1b5",
	"name": "airport_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483331c4952fdf7150e1b6",
	"name": "airport_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483340c4952fdf7150e1b7",
	"name": "city_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48335ac4952fdf7150e1ba",
	"name": "class_type"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483367c4952fdf7150e1bb",
	"name": "connect"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483378c4952fdf7150e1bc",
	"name": "cost_relative"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483383c4952fdf7150e1bd",
	"name": "date_relative"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48338ec4952fdf7150e1be",
	"name": "day_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483399c4952fdf7150e1bf",
	"name": "day_number"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833a9c4952fdf7150e1c2",
	"name": "economy"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833b3c4952fdf7150e1c3",
	"name": "end_time"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833bfc4952fdf7150e1c4",
	"name": "fare_amount"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833cec4952fdf7150e1c6",
	"name": "fare_basis_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833d8c4952fdf7150e1c7",
	"name": "flight_days"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833e2c4952fdf7150e1c8",
	"name": "flight_mod"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833edc4952fdf7150e1c9",
	"name": "flight_number"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4833f8c4952fdf7150e1ca",
	"name": "flight_stop"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483402c4952fdf7150e1cb",
	"name": "flight_time"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48340cc4952fdf7150e1cc",
	"name": "meal"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483417c4952fdf7150e1cd",
	"name": "meal_description"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483422c4952fdf7150e1ce",
	"name": "mod"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48342fc4952fdf7150e1cf",
	"name": "month_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48343ac4952fdf7150e1d0",
	"name": "or"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483446c4952fdf7150e1d1",
	"name": "period_mod"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483451c4952fdf7150e1d2",
	"name": "period_of_day"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48345bc4952fdf7150e1d3",
	"name": "restriction_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483466c4952fdf7150e1d4",
	"name": "round_trip"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483470c4952fdf7150e1d5",
	"name": "start_time"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48347ac4952fdf7150e1d6",
	"name": "state_code"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483484c4952fdf7150e1d7",
	"name": "state_name"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c48348ec4952fdf7150e1d8",
	"name": "time"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c483498c4952fdf7150e1d9",
	"name": "time_relative"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4834a2c4952fdf7150e1da",
	"name": "today_relative"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4834acc4952fdf7150e1db",
	"name": "transport_type"
}, {
	"agent_id": "5c48306bc4952fdf7150e1a9",
	"description": "",
	"id": "5c4834b6c4952fdf7150e1dc",
	"name": "year"
}]
entitymap = {}
for item in entitylist:
  entitymap[item['name']] = item['id']

def auto_rules(query_list):
  d = {}
  for item in query_list:
    result = item['query_slot']
    d.setdefault(result['feature'], [])
    d[result['feature']].append(result)
  count = 0
  rules = []
  for slots, result_list in d.items():
    count += 1
    rule = {
      'name': 'rule' + str(count),
      'nodes': []
    }
    nodes = rule['nodes']
    dropout = [0.0 for _ in range(len(slots) + 1)]
    content = [set() for _ in range(len(slots) + 1)]
    for result in result_list:
      for i in range(len(result['sp'])):
        sp = result['sp'][i]
        if len(result['token'][sp[0]:sp[1]]) > 0:
          content[i].add(' '.join(result['token'][sp[0]:sp[1]]))
        else:
          dropout[i] += 1
    nodes.append({
      'type': 'content',
      'value': ' | '.join(list(content[0])),
      'dropout': dropout[0] / len(result_list)
    })
    for i in range(len(slots)):
      nodes.append({
        'type': 'entity',
        'value': i2s[slots[i]][2:].split('.')[-1],
        'slot': i2s[slots[i]][2:],
        'dropout': 0.0
      })
      nodes.append({
        'type': 'content',
        'value': ' | '.join(list(content[i+1])),
        'dropout': dropout[i+1] / len(result_list)
      })
    rules.append(rule)
  return rules

def make_package(
  entitymap,
  rules,
  name='quantity',
  weight=0.01):
  import json
  package = {
    'name': name,
    'tree': {
      'intent': name,
      'type': 'intent',
      'weight': weight,
      'children': [
        {
          "type": "holder"
        }
      ]
    }
  }
  children = package['tree']['children']
  for rule in rules:
    child = {
      'type': 'order',
      'name': rule['name'],
      'dropout': 0,
      'children': []
    }
    children.append(child)
    nodes = child['children']
    for node in rule['nodes']:
      if node['dropout'] < 1.0:
        if node['type'] == 'content':
          content = node['value'].split(' | ')
          nodes.append({
            'content': content,
            'type': 'content',
            'cut': 0,
            'name': content[0],
            'dropout': node['dropout']
          })
        elif node['type'] == 'entity':
          nodes.append({
            'type': 'content',
            'cut': 0,
            'isSlot': True,
            'name': '<%s>' % node['value'],
            'entity': entitymap[node['value']],
            'slot': node['slot']
          })
  return json.dumps(package)

train = {}
count = 0
with open('data/a_train.txt', 'r', encoding='utf-8') as fin:
  for line in fin:
    count += 1
    query = eval(line)
    train.setdefault(query['intent'], [])
    train[query['intent']].append(query)

for kind in train:
  with open('data/active_packages/%s' % kind, 'w', encoding='utf-8') as fout:
    weight = 1.0 * len(train[kind]) / count
    if weight < 0.01:
      weight = 0.01
    fout.write(make_package(
        entitymap,
        auto_rules(train[kind]),
        name=kind,
        weight=weight))
