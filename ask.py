#!/usr/bin/env python3
"""CLI for BurushaskiRAG:  python3 ask.py "What is the Burushaski word for house?" """
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rag_engine import Index

if len(sys.argv) < 2:
    print(__doc__); raise SystemExit(2)
idx = Index.load(Path(__file__).parent / "data.json")
res = idx.answer(" ".join(sys.argv[1:]))
print(res["answer"])
for c in res["citations"]:
    print(f"  cite: {c['source']} :: {c['heading']} (score {c['score']})")
