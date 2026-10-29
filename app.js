// BurushaskiRAG - browser port of rag_engine.py (identical formulas).
// Retrieval: hybrid BM25 + TF-IDF cosine. Generation: extractive, cited.
// Refusal: if >=40% of question terms appear nowhere in the corpus, refuse.
const STOP = new Set(`a an the is are was were be been being of to in on for with and or
as at by from that this it its i you he she we they what how why when where which
who do does did can could should would will shall may might about into than then
so such no not only also just very`.split(/\s+/));

function tokenize(text) {
  const toks = text.toLowerCase().match(/[a-z0-9][a-z0-9\-']+|[a-z0-9]/g) || [];
  return toks.filter(t => (!STOP.has(t) && t.length > 1) || /^\d+$/.test(t));
}

class RagIndex {
  constructor(chunks) {
    this.chunks = chunks;
    this.n = chunks.length;
    this.docTokens = chunks.map(c => tokenize(c.text + " " + (c.heading || "") + " " + (c.title || "")));
    this.docTF = this.docTokens.map(ts => { const m = new Map(); for (const t of ts) m.set(t, (m.get(t) || 0) + 1); return m; });
    this.docLen = this.docTokens.map(ts => Math.max(1, ts.length));
    this.avgdl = this.docLen.reduce((a, b) => a + b, 0) / Math.max(1, this.n);
    this.df = new Map();
    for (const tf of this.docTF) for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
    this.idf = new Map();
    for (const [t, d] of this.df) this.idf.set(t, Math.log(1 + (this.n - d + 0.5) / (d + 0.5)));
  }
  bm25(qterms, i, k1 = 1.5, b = 0.75) {
    const tf = this.docTF[i], dl = this.docLen[i]; let s = 0;
    for (const t of qterms) if (tf.has(t)) {
      const idf = this.idf.get(t) || 0;
      s += idf * (tf.get(t) * (k1 + 1)) / (tf.get(t) + k1 * (1 - b + b * dl / this.avgdl));
    }
    return s;
  }
  tfidfCos(qtf, i) {
    const tf = this.docTF[i]; let dot = 0, qn = 0, dn = 0;
    for (const [t, qc] of qtf) {
      const idf = this.idf.get(t) || 0, qv = qc * idf;
      qn += qv * qv;
      if (tf.has(t)) dot += qv * (tf.get(t) * idf);
    }
    for (const [t, c] of tf) { const v = c * (this.idf.get(t) || 0); dn += v * v; }
    return (qn === 0 || dn === 0) ? 0 : dot / (Math.sqrt(qn) * Math.sqrt(dn));
  }
  retrieve(query, k = 5) {
    const qterms = tokenize(query);
    if (!qterms.length) return [];
    const qtf = new Map(); for (const t of qterms) qtf.set(t, (qtf.get(t) || 0) + 1);
    const scored = [];
    for (let i = 0; i < this.n; i++) {
      const bm = this.bm25(qterms, i);
      if (bm <= 0) continue;
      const cs = this.tfidfCos(qtf, i);
      scored.push([0.75 * bm + 6.0 * cs, i]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, k).map(([s, i]) => ({ score: Math.round(s * 1000) / 1000, ...this.chunks[i] }));
  }
  answer(query, k = 3) {
    const hits = this.retrieve(query, k);
    const qterms = tokenize(query);
    const uniq = [...new Set(qterms)];
    const oov = uniq.length ? uniq.filter(t => !this.df.has(t)).length / uniq.length : 1;
    if (!hits.length || hits[0].score < 0.6 || oov >= 0.4) {
      return { answer: "Not in this corpus - no chunk scored high enough to answer from. I won't guess.", hits: [], refused: true };
    }
    return { answer: hits[0].text, hits, refused: false };
  }
}

if (typeof module !== "undefined") { module.exports = { RagIndex, tokenize }; }

// ---------- UI ----------
if (typeof document !== "undefined") {
  let idx = null;
  const $ = id => document.getElementById(id);
  fetch("data.json").then(r => r.json()).then(d => {
    idx = new RagIndex(d.chunks);
    $("status").textContent = d.chunks.length + " chunks indexed - ready. Ask below.";
  }).catch(() => { $("status").textContent = "Could not load data.json"; });

  function ask(q) {
    if (!idx || !q.trim()) return;
    const res = idx.answer(q);
    const box = $("result");
    if (res.refused) {
      box.innerHTML = `<div class="refused">${res.answer}</div>`;
      return;
    }
    const [top, ...rest] = res.hits;
    box.innerHTML =
      `<div class="card top"><div class="meta">Best match - ${top.source} :: ${top.heading || ""} (score ${top.score})</div><p>${top.text}</p></div>` +
      (rest.length ? `<div class="more"><h3>Also retrieved</h3>` + rest.map(h =>
        `<div class="card"><div class="meta">${h.source} :: ${h.heading || ""} (score ${h.score})</div><p>${h.text}</p></div>`).join("") + `</div>` : "");
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("go").addEventListener("click", () => ask($("q").value));
    $("q").addEventListener("keydown", e => { if (e.key === "Enter") ask($("q").value); });
    document.querySelectorAll(".chip").forEach(ch => ch.addEventListener("click", () => { $("q").value = ch.textContent; ask(ch.textContent); }));
  });
}
