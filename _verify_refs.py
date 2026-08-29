"""Verify candidate references via the Semantic Scholar Graph API.

For each query we record title, authors, venue, year, DOI. Only entries whose
returned metadata matches the intended paper are kept for references.bib.
"""
import json
import time
import urllib.request
import urllib.parse

CANDIDATES = [
    # MARL / self-play classics
    "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning",
    "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
    "Counterfactual Multi-Agent Policy Gradients",
    "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning",
    "The Mechanics of n-Player Differentiable Games",
    "Learning in games: survey",
    "Learning with Opponent-Learning Awareness",
    "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?",
    "Dota 2 with Large Scale Deep Reinforcement Learning",
    # cognitive radar / RRM
    "Cognitive radar: a way of the future",
    "Cognitive radar framework for target detection and track initiation",
    "Fully adaptive radar for target tracking part I",
    "The Development From Adaptive to Cognitive Radar Resource Management",
    "Cognitive radar for waveform diversity utilization",
    "Spectrum Allocation for Non-Cooperative Radar Coexistence",
    "Adversarial Multi-Player Bandits for Cognitive Radar",
    "A Modified Reinforcement Q-Learning Method for Multi-Function Phased Array Radar Beam Scheduling",
    "Adaptive dwell scheduling based on Q-learning for multifunctional radar",
    # radar anti-jamming DRL
    "Improving anti-jamming decision-making strategies for cognitive radar",
    "A Radar Anti-jamming Method under Multi-Jamming Scenarios Based on Complex-Domain Deep Reinforcement Learning",
    "Design of anti-jamming decision-making for cognitive radar",
    "Game theory and reinforcement learning for anti-jamming defense in wireless communications",
    "Two-Dimensional Anti-Jamming Communication Based on Deep Reinforcement Learning",
    # electronic warfare / phased array
    "Radar Spectrum Engineering and Management: Technical and Regulatory Issues",
]

def fetch(query):
    url = ('https://api.semanticscholar.org/graph/v1/paper/search?'
           + urllib.parse.urlencode({'query': query, 'limit': 1,
                                     'fields': 'title,authors,venue,year,externalIds,publicationVenue,citationCount'}))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.load(r)
        except Exception as e:
            time.sleep(4 * (attempt + 1))
    return {'error': 'failed'}

out = {}
for q in CANDIDATES:
    res = fetch(q)
    if res.get('data'):
        p = res['data'][0]
        out[q] = {
            'title': p.get('title'),
            'authors': [a['name'] for a in (p.get('authors') or [])][:6],
            'venue': p.get('venue'),
            'year': p.get('year'),
            'doi': (p.get('externalIds') or {}).get('DOI'),
            'arxiv': (p.get('externalIds') or {}).get('ArXiv'),
            'citations': p.get('citationCount'),
        }
        print(f"[OK] {q[:60]!r:62} -> {out[q]['year']} | {out[q]['venue'][:50] if out[q]['venue'] else '-'}"
              f" | doi={out[q]['doi']} | cites={out[q]['citations']}")
    else:
        out[q] = {'error': str(res)[:120]}
        print(f"[FAIL] {q[:60]!r:62} -> {out[q]['error']}")
    time.sleep(1.6)

Path = __import__('pathlib').Path
Path('_verified_refs.json').write_text(json.dumps(out, indent=1))
print('\nwrote _verified_refs.json')
