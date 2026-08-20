"""CLI wrapper over RagEngine."""

import sys

from src.rag import RagEngine


def main():
    question = " ".join(sys.argv[1:]) or "How long do I have to notify individuals after a breach?"

    engine = RagEngine()
    result = engine.ask(question)

    print(f"Q: {result.question}\n")
    print(result.answer)

    if result.citations:
        print("\n--- retrieved ---")
        for c in result.citations:
            d = f"{c.distance:.3f}" if c.distance is not None else "bm25"
            print(f"  [{d}] {c.section} {c.title[:50]}")
    else:
        print(f"\n(closest {result.closest_distance:.3f} exceeded {0.75})")


if __name__ == "__main__":
    main()