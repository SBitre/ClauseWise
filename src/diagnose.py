import json
import re
from pathlib import Path

text = Path("data/processed/hipaa_clean.txt").read_text(encoding="utf-8")
chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))

print("=== A: why do Definitions headers fail? ===")
for sec in ["160.103", "164.501"]:
    for m in re.finditer(re.escape(f"{sec}"), text):
        window = text[max(0, m.start() - 15): m.start() + 90]
        print(f"  {sec}: {window!r}")
        break

print("\n=== B: what do the flagged chunks actually start with? ===")
for cid in ["160.518-0", "160.524-0", "164.501-0", "164.506-0"]:
    c = next((c for c in chunks if c["id"] == cid), None)
    if c:
        print(f"  {cid}: {c['text'][:70]!r}")

print("\n=== C: § 160.102 should be short — where does it run to? ===")
c = next(c for c in chunks if c["id"] == "160.102-0")
print(f"  ends with: {c['text'][-200:]!r}")