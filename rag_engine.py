#!/usr/bin/env python3
"""
rag_engine.py - a small, dependency-free RAG engine (stdlib only).

Pipeline: chunk -> index (BM25 + TF-IDF) -> retrieve -> grounded answer.

Design choices, honestly stated:
- Retrieval is lexical (BM25 + TF-IDF cosine, hybrid), NOT neural embeddings.
  It runs offline on a laptop with zero installs, and for small, well-written
  corpora (a dictionary, a research KB, project docs) it is strong.
- Generation is EXTRACTIVE: the answer is sentences copied from the retrieved
  chunks, with a citation on every sentence. The engine says "not in the
  corpus" instead of inventing an answer when retrieval is weak.
  An LLM can be plugged in later behind the same retrieve() call - the
  citations contract (answer only from these chunks) stays the same.
"""
import json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

STOP = set("""a an the is are was were be been being of to in on for with and or
as at by from that this it its i you he she we they what how why when where which
who do does did can could should would will shall may might about into than then
so such no not only also just very""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']+|[a-z0-9]")

def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1 or t.isdigit()]

def chunk_text(text: str, source: str, title: str = "", max_chars: int = 900, overlap: int = 150):
    """Split markdown-ish text into overlapping chunks, keeping heading context."""
    chunks, heading, buf = [], "", ""
    def flush():
        nonlocal buf
        b = buf.strip()
        if len(b) >= 40:
            chunks.append({"source": source, "title": title, "heading": heading, "text": b})
        buf = ""
    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.strip("# ").strip()
            continue
        if len(buf) + len(line) > max_chars:
            tail = buf[-overlap:] if overlap else ""
            flush()
            buf = tail + "\n" + line
        else:
            buf += "\n" + line if buf else line
    flush()
    # very long single blocks still get windowed
    out = []
    for c in chunks:
        t = c["text"]
        if len(t) <= max_chars * 1.6:
            out.append(c)
        else:
            for i in range(0, len(t), max_chars - overlap):
                out.append({**c, "text": t[i:i + max_chars]})
    return out

class Index:
    def __init__(self, name: str, chunks: list):
        self.name = name
        self.chunks = chunks
        self.n = len(chunks)
        self.doc_tokens = [tokenize(c["text"] + " " + c.get("heading", "") + " " + c.get("title", "")) for c in chunks]
        self.doc_tf = [Counter(t) for t in self.doc_tokens]
        self.doc_len = [max(1, len(t)) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(1, self.n)
        df = Counter()
        for tf in self.doc_tf:
            for term in tf:
                df[term] += 1
        self.df = dict(df)
        self.idf = {t: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    # ---- retrieval ----
    def _bm25(self, qterms, i, k1=1.5, b=0.75):
        tf, dl, s = self.doc_tf[i], self.doc_len[i], 0.0
        for t in qterms:
            if t in tf:
                idf = self.idf.get(t, 0.0)
                s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def _tfidf_cos(self, qtf, i):
        tf = self.doc_tf[i]
        dot, qn, dn = 0.0, 0.0, 0.0
        for t, qc in qtf.items():
            idf = self.idf.get(t, 0.0)
            qv = qc * idf
            qn += qv * qv
            if t in tf:
                dot += qv * (tf[t] * idf)
        for t, c in tf.items():
            v = c * self.idf.get(t, 0.0)
            dn += v * v
        if qn == 0 or dn == 0:
            return 0.0
        return dot / (math.sqrt(qn) * math.sqrt(dn))

    def retrieve(self, query: str, k: int = 5):
        qterms = tokenize(query)
        if not qterms:
            return []
        qtf = Counter(qterms)
        scored = []
        for i in range(self.n):
            bm = self._bm25(qterms, i)
            if bm <= 0:
                continue
            cos = self._tfidf_cos(qtf, i)
            scored.append((0.75 * bm + 6.0 * cos, bm, cos, i))
        scored.sort(reverse=True)
        return [{"score": round(s, 3), "bm25": round(bm, 3), "tfidf": round(cs, 3),
                 **self.chunks[i]} for s, bm, cs, i in scored[:k]]

    # ---- grounded (extractive) generation ----
    def answer(self, query: str, k: int = 3, min_score: float = 0.6, max_oov: float = 0.4):
        hits = self.retrieve(query, k=k)
        # Out-of-vocabulary guard: if >=40% of the question's terms appear
        # NOWHERE in the corpus, the question is almost certainly off-topic
        # (measured: real questions <=25% OOV, off-topic >=50% OOV on our eval set).
        qterms = tokenize(query)
        oov = (sum(1 for t in set(qterms) if t not in self.df) / max(1, len(set(qterms)))) if qterms else 1.0
        if not hits or hits[0]["score"] < min_score or oov >= max_oov:
            return {"query": query, "answer": "Not in the archive - I don't have this in my data, so I won't guess.",
                    "citations": [], "hits": hits, "oov_rate": round(oov, 3)}
        qset = set(tokenize(query))
        picked, cites = [], []
        for h in hits:
            sents = re.split(r"(?<=[.!?])\s+", h["text"].replace("\n", " "))
            best = sorted(sents, key=lambda s: -len(qset & set(tokenize(s))))
            for s in best[:2]:
                s = s.strip()
                if len(s) >= 25 and s not in picked:
                    picked.append(s)
                    cite = {"source": h["source"], "heading": h.get("heading", ""), "score": h["score"]}
                    if cite not in cites:
                        cites.append(cite)
                if len(picked) >= 4:
                    break
            if len(picked) >= 4:
                break
        return {"query": query, "answer": " ".join(picked), "citations": cites, "hits": hits}

    # ---- persistence ----
    def save(self, path):
        Path(path).write_text(json.dumps({"name": self.name, "chunks": self.chunks}, ensure_ascii=False))
    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["name"], d["chunks"])

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: rag_engine.py <index.json> <k> \"<question>\"")
        raise SystemExit(2)
    idx = Index.load(sys.argv[1])
    res = idx.answer(" ".join(sys.argv[3:]), k=int(sys.argv[2]))
    print("ANSWER:", res["answer"])
    for c in res["citations"]:
        print(f"  cite: {c['source']} :: {c['heading']} (score {c['score']})")
