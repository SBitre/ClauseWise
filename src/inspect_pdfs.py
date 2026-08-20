import re
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")

for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    print("=" * 70)
    print(f"FILE: {pdf_path.name}")
    print(f"PAGES: {len(reader.pages)}")

    full_text = "\n".join(
        page.extract_text() or "" for page in reader.pages
    )
    print(f"CHARACTERS: {len(full_text):,}")

    # How many section markers can we find?
    sections = re.findall(r"§+\s*\d+\.\d+", full_text)
    print(f"SECTION MARKERS FOUND: {len(sections)}")
    print(f"FIRST FEW: {sections[:8]}")

    # What does the raw text actually look like?
    mid = len(full_text) // 2
    print("\n--- SAMPLE FROM MIDDLE ---")
    print(full_text[mid:mid + 1200])
    print()