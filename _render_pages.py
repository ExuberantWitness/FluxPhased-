import fitz  # PyMuPDF
from pathlib import Path

paper = Path(r'E:\DATA\vscode\FluxPhased\paper')
out = paper / 'visual_pages'
out.mkdir(exist_ok=True)
doc = fitz.open(paper / 'main.pdf')
for i, page in enumerate(doc, 1):
    pix = page.get_pixmap(dpi=110)
    pix.save(out / f'page-{i}.png')
    print(f'page-{i}.png {pix.width}x{pix.height}')
doc.close()
