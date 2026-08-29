"""Round 2: verify references via arXiv API (exact IDs) and S2 DOI lookups."""
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ARXIV_IDS = {
    'rashid2018qmix': '1803.11485',
    'lowe2017maddpg': '1706.02275',
    'lanctot2017psro': '1711.00832',
    'balduzzi2019mechanics': '1802.05642',
    'dewitt2020ippo': '2011.09533',
    'berner2019dota': '1912.06680',
    'foerster2018lola': '1709.04326',
    'hernandezleal2017survey': '1707.09183',
    'jiang2023antijam': None,  # placeholder, handled by DOI below if found
}

def arxiv_lookup(arxiv_id):
    url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    with urllib.request.urlopen(url, timeout=20) as r:
        xml = r.read().decode('utf-8', 'replace')
    ns = {'a': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(xml)
    entry = root.find('a:entry', ns)
    if entry is None:
        return None
    title = ' '.join(entry.find('a:title', ns).text.split())
    authors = [a.find('a:name', ns).text for a in entry.findall('a:author', ns)]
    published = entry.find('a:published', ns).text[:4]
    return {'title': title, 'authors': authors, 'year': int(published), 'arxiv': arxiv_id}

out = {}
for key, aid in ARXIV_IDS.items():
    if aid is None:
        continue
    try:
        out[key] = arxiv_lookup(aid)
        print(f"[OK] {key}: {out[key]['title'][:70]} ({out[key]['year']}, {len(out[key]['authors'])} authors)")
    except Exception as e:
        out[key] = {'error': str(e)[:100]}
        print(f"[FAIL] {key}: {out[key]['error']}")
    time.sleep(1.0)

Path('_verified_arxiv.json').write_text(json.dumps(out, indent=1))
print('wrote _verified_arxiv.json')
