import json
from pathlib import Path

chunks = json.loads(
    (Path(__file__).resolve().parent.parent / "data/processed/chunks.json")
    .read_text(encoding="utf-8")
)

bad = [
    c for c in chunks
    if c["id"].endswith("-0")
    and not c["text"].lstrip().startswith(f"§ {c['id'].rsplit('-', 1)[0]}")
]

print(f"{len(bad)} of {len(chunks)} chunks mislabeled ({len(bad)/len(chunks):.1%})\n")
for c in bad:
    print(f"{c['id']}  claims: {c['title']}")
    print(f"  starts: {c['text'][:110]!r}\n")