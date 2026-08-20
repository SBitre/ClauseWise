import re
from pathlib import Path

text = Path("data/processed/hipaa_clean.txt").read_text(encoding="utf-8")

SECTION_RE = re.compile(
    r"§\s*(\d{3}\.\d{3,4})\s+"
    r"([A-Z][^.§]{3,140}?)\."
    r"(?!\s*§)"
    r"(?=\s+(?:\([a-z0-9]\)|[A-Z]))"
)

# 1. Every literal occurrence of 160.103 in the file
print("=== All occurrences of '160.103' ===")
for i, m in enumerate(re.finditer(r"160\.103", text)):
    print(f"  [{i}] char {m.start():,}: {text[m.start()-12 : m.start()+110]!r}")

# 2. Test each regex component against the real header
print("\n=== Component tests on § 160.103 ===")
occurrences = [m.start() for m in re.finditer(r"160\.103", text)]
for pos in occurrences:
    window = text[pos - 5 : pos + 200]
    tests = {
        "full regex":        SECTION_RE.search(window),
        "no lookahead":      re.search(r"§\s*(\d{3}\.\d{3,4})\s+([A-Z][^.§]{3,140}?)\.", window),
        "title chars only":  re.search(r"§\s*\d{3}\.\d{3,4}\s+[A-Z][^.§]{3,140}?\.", window),
    }
    print(f"\n  at char {pos:,}: {window[:60]!r}")
    for name, res in tests.items():
        print(f"    {name:<18} -> {res.group()[:60]!r}" if res else f"    {name:<18} -> NO MATCH")

# 3. What did the regex actually find near there?
print("\n=== Regex matches between chars 3,000 and 12,000 ===")
for m in SECTION_RE.finditer(text[3000:12000]):
    print(f"  {m.group(1)}  {m.group(2)[:50]!r}")