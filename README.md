# A vector database from scratch, in ~500 lines of Python

You use a vector database every day — every RAG pipeline, every "chat with
your docs" app, every semantic-search box. Ask an engineer *what it actually
does* and you'll usually get: "it, uh, finds similar vectors. Fast."

This repo is the missing explanation. One file, [`vectordb.py`](vectordb.py)
(466 lines, roughly half of them comments), containing a real, working vector
database:

```python
from vectordb import VectorDB, embed

db = VectorDB(dim=64)
db.upsert("fern-care", embed("water your fern twice a week"), {"topic": "plants"})
db.query(embed("how do I keep my plant alive?"), k=3, where={"topic": "plants"})
```

No numpy. No dependencies at all — even the demo embeddings are built from
scratch. It's 100–1000× slower than faiss, **on purpose**: speed hides ideas,
and this file is the ideas.

## The chapters

The file reads top-to-bottom like a short book:

| Chapter | What you'll finally understand |
|---|---|
| 1 — Similarity | Why cosine, and the normalize-once trick every real DB uses |
| 2 — Exact search | Brute force: 100% correct, O(N·d), the problem statement |
| 3 — HNSW | The graph index inside Qdrant/Weaviate/pgvector/Milvus, in ~150 lines |
| 4 — The shell | IDs, metadata, filters — and why HNSW *cannot delete* (tombstones!) |
| 5 — Persistence | The whole database as one JSON file you can `cat` |

## The tests are the tutorial

[`test_vectordb.py`](test_vectordb.py) restates each chapter's claims as
assertions — including measuring HNSW's recall@10 against brute-force ground
truth and *counting distance computations* to prove the graph does less work:

```bash
python3 vectordb.py        # 30-second demo, zero deps
python3 test_vectordb.py   # the guided tour (pytest works too)
```

## FAQ

**Should I use this in production?** No. Use Qdrant, pgvector, or faiss.
Use *this* to understand what those are doing.

**Why is it slow?** Pure-Python floats, one multiply at a time. Real engines
do the *same math* with float32 arrays and SIMD. Nothing conceptual differs.

**Is this real HNSW?** Yes — layered graph, geometric level assignment,
beam search with early exit, bidirectional linking with pruning. Omitted:
the diversity-selection heuristic (§4 of the paper) — a perfect
extra chapter, see below.

## Contributing

Two kinds of contributions, both very welcome:

**Translations.** Translate the comments (code stays identical — that's the
contract) into your language under `translations/<lang>/vectordb.py`, or start
with just `translations/<lang>/README.md`. Tiếng Việt is seeded first in
[`translations/vi/`](translations/vi/). Verify with:
`python3 tools/check_translation.py translations/<lang>/vectordb.py`
(strips comments from both files and diffs the code).

**Extra chapters.** Self-contained annotated files in `extras/`, same rules
(stdlib-only, heavily commented, tests that teach). Shipped so far:

- [Chapter 6 — IVF](extras/06_ivf.py): k-means partitioning + `nprobe`, the
  *other* big ANN family — and the index that can actually delete
- [Chapter 7 — Product quantization](extras/07_pq.py): 16× compression, and
  the ADC trick that searches the bytes without decompressing them

Wishlist:
- HNSW's neighbor-diversity heuristic (§4 of the paper) and why it matters
- Pre- vs post-filtering: filtered ANN done properly
- Binary quantization + rescoring
- A write-ahead log: crash-safe persistence
- Benchmarks: this repo vs faiss on real embeddings, recall/QPS curves

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions.

## License

MIT.
