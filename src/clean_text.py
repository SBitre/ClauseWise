import re
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("data")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_TEXT = "HIPAA Administrative Simplification Regulation Text"
DATE_LINE = re.compile(r"^[A-Z][a-z]+ \d{4}$")
PAGE_NUM = re.compile(r"^\d{1,3}$")
FR_CITE = re.compile(r"\[\d+\s*FR\s*\d+[^\]]*\]")
DOT_LEADER = re.compile(r"\.{5,}")


def clean_page(raw: str) -> str:
    """Drop header/footer noise, then unwrap the narrow-column line breaks."""
    kept = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if HEADER_TEXT in s or DATE_LINE.match(s) or PAGE_NUM.match(s):
            continue
        kept.append(s)

    text = ""
    for s in kept:
        if not text:
            text = s
        elif text.endswith("-"):          # hyphenated word split across lines
            text = text[:-1] + s
        else:
            text += " " + s
    return text


def main():
    pdf_path = DATA_DIR / "hipaa-simplification-201303.pdf"
    reader = PdfReader(pdf_path)

    pages, toc_pages = [], []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if len(DOT_LEADER.findall(raw)) > 5:   # table-of-contents page
            toc_pages.append(i)
        pages.append(clean_page(raw))

    body_start = max(toc_pages) + 1 if toc_pages else 0
    text = " ".join(p for p in pages[body_start:] if p)
    text = FR_CITE.sub("", text)            # strip amendment history
    text = re.sub(r"\s{2,}", " ", text).strip()

    out_path = OUT_DIR / "hipaa_clean.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"TOC pages detected : {len(toc_pages)} (body starts at page {body_start})")
    print(f"Characters kept    : {len(text):,}")
    print(f"Footer leftovers   : {text.count(HEADER_TEXT)}  <- must be 0")
    print(f"Section markers    : {len(re.findall(r'§+\\s*\\d+\\.\\d+', text))}")
    print(f"'electronic'       : {text.count('electronic')}")
    print(f"'elec tronic'      : {text.count('elec tronic')}  <- word-split damage")
    print(f"\nSaved to {out_path}")

    idx = text.find("§ 164.308")
    print("\n--- SAMPLE AT § 164.308 ---")
    print(text[idx:idx + 900])


if __name__ == "__main__":
    main()