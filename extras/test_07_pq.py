"""
test_07_pq.py — the guided tour for extra chapter 7 (Product Quantization).

Claims from 07_pq.py, proven by assertion. Runs standalone or under pytest.
"""

import heapq
import importlib.util
import pathlib
import random

# Numbered chapter files load by path (see test_06_ivf.py for why).
_spec = importlib.util.spec_from_file_location(
    "pq", pathlib.Path(__file__).parent / "07_pq.py")
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)

N, DIM, M, K = 1200, 16, 4, 32
_rng = random.Random(42)
DATASET = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
QUERIES = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(20)]

_cache = {}


def build() -> "pq.PQIndex":
    if "idx" not in _cache:
        idx = pq.PQIndex(dim=DIM, m=M, k=K)
        for v in DATASET:
            idx.add(v)
        idx.train()
        _cache["idx"] = idx
    return _cache["idx"]


def true_top10(q):
    qn = pq.normalize(q)
    return {i for _, i in heapq.nlargest(
        10, ((pq.dot(qn, pq.normalize(v)), i)
             for i, v in enumerate(DATASET)))}


def recall_at_10(idx, rerank):
    return sum(len(true_top10(q) &
                   {i for _, i in idx.search(q, 10, rerank=rerank)}) / 10
               for q in QUERIES) / len(QUERIES)


def sqerr(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def test_ch7_codes_are_tiny_integers():
    """The compression is real and inspectable: every vector became m
    integers in [0, k), and each of the m codebooks holds k codewords of
    dim/m floats — the only floats the 'RAM tier' ever keeps."""
    idx = build()
    assert all(len(code) == M and all(0 <= c < K for c in code)
               for code in idx._codes)
    assert len(idx._books) == M
    assert all(len(book) == K and len(book[0]) == DIM // M
               for book in idx._books)


def test_ch7_sixteen_x_smaller():
    """The headline arithmetic: 16 float32 dims (64 bytes) collapse to 4
    one-byte codes. Same ratio that turns 61 GB of OpenAI embeddings into
    under 4 GB."""
    raw, packed = build().bytes_per_vector()
    assert (raw, packed) == (64, 4)


def test_ch7_decode_recovers_the_gist():
    """Quantization is lossy but not amnesiac: a vector's own
    reconstruction sits far closer to it than the reconstruction of a
    random other vector — the codes genuinely remember who they were."""
    idx = build()
    own = sum(sqerr(pq.normalize(DATASET[i]), idx.decode(i))
              for i in range(200)) / 200
    other = sum(sqerr(pq.normalize(DATASET[i]), idx.decode(i + 200))
                for i in range(200)) / 200
    assert own < other * 0.5
    assert own < 0.6          # unit vectors: max possible sqerr is 4.0


def test_ch7_richer_codebooks_forget_less():
    """The size/fidelity dial: k=32 codewords reconstruct strictly better
    than k=4 — you buy accuracy with codebook bits (and at k=N you'd have
    memorized the dataset)."""
    errs = {}
    for k in (4, 32):
        idx = pq.PQIndex(dim=DIM, m=M, k=k)
        for v in DATASET[:300]:
            idx.add(v)
        idx.train()
        errs[k] = sum(sqerr(pq.normalize(DATASET[i]), idx.decode(i))
                      for i in range(300)) / 300
    assert errs[32] < errs[4]


def test_ch7_adc_scores_track_true_scores():
    """ADC never touches the stored vectors, only lookup tables — yet its
    scores stay close to the true dot products (small mean error), which
    is the entire license for searching without decompressing."""
    idx = build()
    q = pq.normalize(QUERIES[0])
    table = [[pq.dot(chunk, w) for w in idx._books[j]]
             for j, chunk in enumerate(idx._chunks(q))]
    err = sum(abs(sum(table[j][cj] for j, cj in enumerate(idx._codes[i]))
                  - pq.dot(q, pq.normalize(DATASET[i])))
              for i in range(300)) / 300
    assert err < 0.15, f"mean ADC error {err:.3f}"


def test_ch7_rerank_equal_to_topk_is_useless():
    """The shortlist subtlety, asserted: rescoring exactly the topk you
    were going to return reorders them but recovers nothing — the same
    ids come back. Rerank only earns its keep when it digs DEEPER than k."""
    idx = build()
    q = QUERIES[1]
    plain = {i for _, i in idx.search(q, 10)}
    same = {i for _, i in idx.search(q, 10, rerank=10)}
    assert plain == same


def test_ch7_bytes_shortlist_floats_decide():
    """The production pattern, end to end: byte-only recall is mediocre
    (that's the compression tax), but letting the bytes nominate ~8% of
    the corpus and full floats re-judge them recovers most of the truth —
    RAM does the scan, disk-sized truth only handles the shortlist."""
    idx = build()
    alone = recall_at_10(idx, rerank=0)
    refined = recall_at_10(idx, rerank=100)
    assert alone >= 0.25                     # bytes alone: usable, not great
    assert refined >= 0.75                   # shortlist + rescore: strong
    assert refined >= alone + 0.20           # and the gap is the point


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed.")
