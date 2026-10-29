# BurushaskiRAG

**Live demo: https://aazimameen1122.github.io/burushaski-rag/**

A small retrieval-augmented generation (RAG) system over the
[Burushaski Digital Archive](https://github.com/aazimameen1-dotcom/Burushaski-Digital-Archive).
Ask it a question in plain English; it answers from the archive itself,
names its source on every result, and refuses when the answer is not in
the corpus instead of guessing.

Burushaski is a language isolate spoken in Hunza, Nagar and Yasin
(Gilgit-Baltistan), with a small, endangered enclave in Srinagar, Kashmir.
This RAG covers both: the dictionary and the Srinagar community research.

## Corpus (94 chunks)
- 62 dictionary entries - Srinagar + Gilgit dialects, Perso-Arabic script,
  IPA, and an example sentence per dialect
- 4 audio archive recordings (folktale, devotional, two folk songs)
- Archive README + the Srinagar Burushaski research report
  (history and origins, demographics, dialect features, endangerment)

## How it works
One engine, two runtimes with identical rankings:
- `rag_engine.py` - Python, stdlib only. Heading-aware chunking, hybrid
  BM25 + TF-IDF retrieval, extractive cited answers, and an
  out-of-vocabulary refusal guard (>=40% unseen terms -> refuse).
- `app.js` - the same engine in JavaScript, running fully in the browser
  on the GitHub Pages demo. No server, no API key, no tracking.

## Run the Python version
```bash
python3 ask.py "What is the Burushaski word for house?"
python3 ask.py "When did the princes of Hunza and Nagar migrate to Kashmir?"
python3 ask.py "What is the capital of France?"   # -> refuses, correctly
```

## Measured results
Labelled test set (5 positive + 1 off-topic question on this corpus,
part of an 18-question suite across 3 sibling RAG systems):
Hit@1 **1.00**, Hit@3 **1.00**, MRR **1.00**, off-topic refused 1/1.
Small set - a smoke test, not a benchmark.

## Honest limits
- Retrieval is lexical (BM25 + TF-IDF), not neural embeddings; synonyms
  the corpus never uses can miss.
- Answers are extractive, so they can read choppy. An LLM behind the same
  retrieve-and-cite contract is the planned next step.

Part of a set of three RAG systems built 29 Oct 2026 (BurushaskiRAG,
DermaRAG, PortfolioRAG) by Aazim Ameen, DYoD student at IUST, Kashmir.
