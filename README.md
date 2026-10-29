# BurushaskiRAG

**Retrieval-augmented search over the Burushaski Digital Archive — answers from the corpus, citations on every result, and a refusal when the corpus does not know.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-c96a17)](https://aazimameen1122.github.io/burushaski-rag/)
[![Python](https://img.shields.io/badge/Python-3.x%20·%20stdlib%20only-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Retrieval Hit@1](https://img.shields.io/badge/Retrieval%20Hit%401-1.00-2e9e5b)](#evaluation)
[![API Keys](https://img.shields.io/badge/API%20keys-none-43534b)](#architecture)

### ▶ Live demo: **https://aazimameen1122.github.io/burushaski-rag/**

Ask in plain English — *"What is the Burushaski word for house?"*, *"When did the princes of Hunza and Nagar migrate to Kashmir?"* — and get the archive's own text back, with its source. Ask something outside the archive and the system declines: *"Not in this corpus. I won't guess."*

---

## Overview

[Burushaski](https://en.wikipedia.org/wiki/Burushaski) is a language isolate spoken in Hunza, Nagar and Yasin (Gilgit-Baltistan), with a small, endangered enclave in Srinagar, Kashmir. The [Burushaski Digital Archive](https://github.com/aazimameen1-dotcom/Burushaski-Digital-Archive) documents the Srinagar variety. **BurushaskiRAG makes that archive answerable**: one small retrieval engine over the dictionary, the recordings catalogue, and the community research report.

It is deliberately built without the usual stack — no embeddings API, no vector database, no LLM, no server. The same engine runs in Python and, identically, in the browser on the live demo.

## Features

- **Dialect-aware lookup** — dictionary answers carry both Srinagar and Gilgit forms, Perso-Arabic script, IPA, and an example sentence per dialect
- **Cited answers** — every result names its source file, section and retrieval score
- **Hallucination guard** — an out-of-vocabulary refusal rule (≥ 40% unseen terms) makes off-corpus questions fail closed, not creatively
- **Zero infrastructure** — static site + one JSON index; the Python version needs nothing beyond the standard library
- **Measured, not asserted** — retrieval quality is evaluated on a labelled question set (see below)

## Architecture

```
        corpus                          engine                         answer
 ┌───────────────────┐        ┌──────────────────────────┐        ┌────────────────────┐
 │ data.js dictionary │        │ chunk (heading-aware)    │        │ extractive answer  │
 │ recordings         │ ─────▶ │ index (BM25 + TF-IDF)    │ ─────▶ │ + source citation  │
 │ README · report    │        │ retrieve (hybrid score)  │        │ or a refusal       │
 └───────────────────┘        │ refuse (OOV guard)       │        └────────────────────┘
                               └──────────────────────────┘
```

| Stage | Implementation |
|---|---|
| Chunking | Heading-aware, ~900 chars with 150-char overlap; one chunk per dictionary entry |
| Retrieval | Hybrid score = 0.75 × BM25 (k1 = 1.5, b = 0.75) + 6.0 × TF-IDF cosine |
| Generation | Extractive — answer text is taken from retrieved chunks, never freely generated |
| Refusal | Question with ≥ 40% out-of-vocabulary terms, or no chunk above threshold, returns a refusal |
| Runtimes | `rag_engine.py` (Python, stdlib only) and `app.js` (browser) — rankings verified identical |

## Corpus

| Source | Content | Chunks |
|---|---|---|
| `Burushaski-Digital-Archive/data.js` | 62 dictionary entries (Srinagar + Gilgit, script, IPA, examples) | 62 |
| `Burushaski-Digital-Archive/data.js` | 4 recordings: folktale, devotional, two folk songs | 4 |
| Archive README + Srinagar research report | Language background, history & origins, demographics, dialect features, endangerment | 28 |
| **Total** | | **94** |

## Quickstart

**Browser** — open the [live demo](https://aazimameen1122.github.io/burushaski-rag/). Nothing to install.

**Python**

```bash
git clone https://github.com/aazimameen1122/burushaski-rag.git
cd burushaski-rag
python3 ask.py "What is the Burushaski word for house?"
# Burushaski word: ha (part of speech: noun). Meaning in English: house. ...
#   cite: Burushaski-Digital-Archive/data.js :: ha (score 7.223)

python3 ask.py "What is the capital of France?"
# Not in this corpus - no chunk scored high enough to answer from. I won't guess.
```

## Evaluation

Labelled set for this corpus: 5 positive questions + 1 off-topic question (part of an 18-question suite across three sibling RAG systems on the same engine).

| Metric | Score |
|---|---|
| Hit@1 | **1.00** |
| Hit@3 | **1.00** |
| MRR | **1.00** |
| Off-topic correctly refused | 1 / 1 |
| Latency | Full suite < 1 s, offline |

A small set — a smoke test, not a benchmark. The refusal threshold was calibrated on it: real questions measured ≤ 25% out-of-vocabulary terms; off-topic questions ≥ 50%.

## Project structure

```
burushaski-rag/
├── index.html       # Live demo page (GitHub Pages)
├── app.js           # RAG engine + UI, JavaScript port
├── data.json        # The index: 94 chunks with sources
├── rag_engine.py    # RAG engine, Python (stdlib only)
├── ask.py           # Command-line interface
└── README.md
```

## Limitations

- **Lexical retrieval.** BM25 + TF-IDF cannot match synonyms the corpus never uses; neural embeddings would recover some of those misses.
- **Extractive answers.** Faithful by construction, plain in style. Plugging a language model behind the same retrieve-and-cite contract is the planned next step.
- **Corpus size.** 62 dictionary entries is the verified archive as it stands; the RAG improves as the archive grows.

## Roadmap

- [ ] LLM generation layer behind the existing citation + refusal contract
- [ ] Grow the index with new verified dictionary entries and elder narratives
- [ ] Semantic (embedding) retrieval as an optional second ranker

## Author

**Aazim Ameen** — photographer and applied-AI builder from Kashmir; DYoD student at IUST Awantipora.
BurushaskiRAG is one of three RAG systems built on this engine on 29 Oct 2026 (BurushaskiRAG · DermaRAG · PortfolioRAG).

Source data: [Burushaski Digital Archive](https://github.com/aazimameen1-dotcom/Burushaski-Digital-Archive)
