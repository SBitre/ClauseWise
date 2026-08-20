import re
from pathlib import Path

text = Path("data/processed/hipaa_clean.txt").read_text(encoding="utf-8")

pat = re.compile(r"§+\s*\d+\.\d+")
hits = list(pat.finditer(text))
print(f"Section markers (fixed regex): {len(hits)}\n")

# A real provision has substantial prose before the next § marker.
print("First 25 markers and the gap to the next one:")
for i, m in enumerate(hits[:25]):
    nxt = hits[i + 1].start() if i + 1 < len(hits) else len(text)
    gap = nxt - m.start()
    flag = "BODY" if gap > 400 else "toc "
    print(f"  {flag}  {m.group():<12} gap={gap:>6}")

# Where does sustained body text begin?
for i, m in enumerate(hits):
    nxt = hits[i + 1].start() if i + 1 < len(hits) else len(text)
    if nxt - m.start() > 800:
        print(f"\nFirst substantial section: {m.group()} at char {m.start():,}")
        print(f"({m.start() / len(text):.1%} into the document)")
        print("\n--- SAMPLE ---")
        print(text[m.start():m.start() + 700])
        break