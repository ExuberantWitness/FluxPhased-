"""Round 3: verify radar-side references via S2 DOI lookup + paced title search."""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE = 'https://api.semanticscholar.org/graph/v1/paper/'
FIELDS = 'title,authors,venue,year,externalIds,citationCount'

DOIS = {
    'haykin2006cognitive': '10.1109/MSP.2006.1593335',
    'xiao2017twodim': '10.1109/ICASSP.2017.7952524',
    'xie2023multijam': '10.12000/JR23139',
    'wang2024ietrsn': '10.1049/rsn2.12497',
    'jsee2025dwell': '10.23919/JSEE.2025.000111',
    'bell2014far': '10.1109/RADAR.2014.6875604',
    'charlish2020development': '10.1109/MAES.2019.2957847',
    'tracy2022bandits': '10.1109/RadarConf2248738.2022.9764226',
    'coma2017foerster': '10.1609/aaai.v32i1.11794',
}

TITLES = {
    'jiang2023dsp': 'Improving anti-jamming decision-making strategies for cognitive radar',
    'jia2024survey': 'Game theory and reinforcement learning for anti-jamming defense in wireless communications',
    'martone2021waveform': 'Cognitive radar for waveform diversity utilization',
    'qlearn2023beam': 'A Modified Reinforcement Q-Learning Method for Multi-Function Phased Array Radar',
    'bell2015jstsp': 'Cognitive radar framework for target detection and tracking',
    'martone2018spectrum': 'Spectrum Allocation for Non-Cooperative Radar Coexistence',
}

out = {}

def get(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ref-verify/1.0'})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))

for key, doi in DOIS.items():
    try:
        p = get(BASE + 'DOI:' + urllib.parse.quote(doi) + '?fields=' + FIELDS)
        out[key] = {'doi': doi, 'title': p.get('title'), 'venue': p.get('venue'),
                    'year': p.get('year'),
                    'authors': [a['name'] for a in (p.get('authors') or [])][:8],
                    'citations': p.get('citationCount')}
        print(f"[OK] {key}: {out[key]['year']} | {out[key]['venue'][:55]} | cites={out[key]['citations']}")
    except Exception as e:
        out[key] = {'doi': doi, 'error': str(e)[:100]}
        print(f"[FAIL] {key}: {out[key]['error']}")
    time.sleep(6)

for key, title in TITLES.items():
    try:
        url = (BASE + 'search?' + urllib.parse.urlencode(
            {'query': title, 'limit': 1, 'fields': FIELDS}))
        res = get(url)
        if res.get('data'):
            p = res['data'][0]
            out[key] = {'title': p.get('title'), 'venue': p.get('venue'),
                        'year': p.get('year'),
                        'doi': (p.get('externalIds') or {}).get('DOI'),
                        'authors': [a['name'] for a in (p.get('authors') or [])][:8],
                        'citations': p.get('citationCount')}
            print(f"[OK] {key}: {out[key]['year']} | {str(out[key]['venue'])[:55]} | doi={out[key]['doi']}")
        else:
            out[key] = {'query': title, 'error': 'no results'}
            print(f"[MISS] {key}")
    except Exception as e:
        out[key] = {'query': title, 'error': str(e)[:100]}
        print(f"[FAIL] {key}: {out[key]['error']}")
    time.sleep(6)

Path('_verified_radar_refs.json').write_text(json.dumps(out, indent=1))
print('wrote _verified_radar_refs.json')
