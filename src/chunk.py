import json
import re
from collections import Counter
from pathlib import Path

SRC = Path("data/processed/hipaa_clean.txt")
OUT = Path("data/processed/chunks.json")

MIN_SECTION_CHARS = 300
MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 200

SECTION_RE = re.compile(
    r"§\s*(\d{3}\.\d{3,4})\s+"          # § 164.312
    r"([A-Z][^.§]{3,140}?)\."           # Title Case, ending in a period
    r"(?=\s+(?:\([a-z0-9]\)|[A-Z]))"    # real body follows
)


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=\.) (?=[A-Z(§])", text)
    parts, cur = [], ""
    for s in sentences:
        while len(s) > MAX_CHUNK_CHARS:
            parts.append(s[:MAX_CHUNK_CHARS])
            s = s[MAX_CHUNK_CHARS - OVERLAP_CHARS:]
        if len(cur) + len(s) > MAX_CHUNK_CHARS and cur:
            parts.append(cur.strip())
            cur = cur[-OVERLAP_CHARS:] + " " + s
        else:
            cur += " " + s
    if cur.strip():
        parts.append(cur.strip())
    return parts


def main():
    text = SRC.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(text))

    # Collect every candidate occurrence of each section.
    candidates: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    skipped = 0

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if len(body) < MIN_SECTION_CHARS:
            skipped += 1
            continue

        number = m.group(1)
        title = m.group(2).strip().rstrip(".")
        if number not in candidates:
            candidates[number] = []
            order.append(number)
        candidates[number].append((title, body))

    # A section number can match twice: once in a subpart contents listing, once
    # as the real provision. The real one carries far more body text.
    duplicates = {n: len(v) for n, v in candidates.items() if len(v) > 1}
    resolved = {
        n: max(v, key=lambda tb: len(tb[1]))
        for n, v in candidates.items()
    }

    chunks = []
    for number in order:
        title, body = resolved[number]
        for j, piece in enumerate(split_long(body)):
            chunks.append({
                "id": f"{number}-{j}",
                "section": f"§ {number}",
                "title": title,
                "part": number.split(".")[0],
                "text": piece,
                "chars": len(piece),
            })

    # Hard guarantee: Chroma requires unique ids.
    ids = [c["id"] for c in chunks]
    dupe_ids = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupe_ids, f"Duplicate chunk ids: {dupe_ids}"

    OUT.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    sizes = [c["chars"] for c in chunks]
    print(f"Section matches       : {len(matches)}")
    print(f"Skipped as TOC/stub   : {skipped}")
    print(f"Duplicate sections    : {len(duplicates)} resolved by longest body")
    if duplicates:
        print(f"    {sorted(duplicates)[:8]}")
    print(f"Unique sections       : {len(resolved)}")
    print(f"Chunks created        : {len(chunks)}")
    print(f"Unique ids            : {len(set(ids))}  <- must equal chunks")
    print(f"Size min/avg/max      : {min(sizes)} / {sum(sizes)//len(sizes)} / {max(sizes)}")
    print(f"Parts covered         : {sorted({c['part'] for c in chunks})}")

    bad = [c for c in chunks if "," in c["title"] or "§" in c["title"]]
    print(f"Suspicious titles     : {len(bad)}  <- must be 0")

    counts = Counter(c["section"] for c in chunks)
    print("\nMost-split sections:")
    for sec, n in counts.most_common(5):
        print(f"    {sec:<12} {n} pieces   {next(c['title'] for c in chunks if c['section'] == sec)}")

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()