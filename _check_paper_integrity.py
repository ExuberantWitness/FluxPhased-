from pathlib import Path
import re
paper = Path(r'E:\DATA\vscode\FluxPhased\paper')
text = '\n'.join(p.read_text(errors='ignore') for p in [paper/'main.tex', *sorted((paper/'sections').glob('*.tex'))])
keys = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', text):
    keys.update(k.strip() for k in m.group(1).split(',') if k.strip())
bib = (paper/'references.bib').read_text(errors='ignore')
bib_keys = set(re.findall(r'^@\w+\{([^,]+),', bib, flags=re.M))
print('CITED', sorted(keys))
print('BIB', sorted(bib_keys))
print('MISSING', sorted(keys - bib_keys))
print('UNCITED', sorted(bib_keys - keys))
for token in ['63.7%\\pm0.7%', 'three-seed S6', 'pre-registered', 'remains unchanged']:
    print('STALE', token, token in text)
assert not (keys - bib_keys)
assert not (bib_keys - keys)
