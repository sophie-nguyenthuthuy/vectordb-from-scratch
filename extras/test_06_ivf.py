"""
test_06_ivf.py — the guided tour for extra chapter 6 (IVF).

Same format as the main suite: every test docstring is a claim from
06_ivf.py, and the assertion proves it. Runs standalone or under pytest.
"""

import heapq
import importlib.util
import pathlib
import random

# A module named `06_ivf` can't be imported with a plain `import` (names
# can't start with a digit) — load it by path instead. Extras keep numeric
# prefixes because reading order matters more than importability.
_spec = importlib.util.spec_from_file_location(
    "ivf", pathlib.Path(__file__).parent / "06_ivf.py")
ivf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ivf)

# Shared corpus, same recipe as the main suite: uniform directions on the
# sphere — the least clusterable, most honest data you can hand k-means.
N, DIM, NLIST = 1500, 16, 32
_rng = random.Random(42)
DATASET = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
QUERIES = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(20)]

_cache = {}


def build() -> "ivf.IVFIndex":
    if "idx" not in _cache:
        idx = ivf.IVFIndex(dim=DIM, nlist=NLIST)
        for v in DATASET:
            idx.add(v)
        idx.train()
        _cache["idx"] = idx
    return _cache["idx"]


def true_top10(q):
    """Ground truth by exhaustive scan — this chapter's own 12-line
    Chapter 2, so the file needs nothing outside itself."""
    qn = ivf.normalize(q)
    return {i for _, i in heapq.nlargest(
        10, ((ivf.dot(qn, ivf.normalize(v)), i)
             for i, v in enumerate(DATASET)))}


def recall_at_10(idx, nprobe):
    return sum(len(true_top10(q) &
                   {i for _, i in idx.search(q, 10, nprobe=nprobe)}) / 10
               for q in QUERIES) / len(QUERIES)


def test_ch6_kmeans_files_every_vector_on_exactly_one_shelf():
    """Training partitions the library: the inverted lists cover all N
    vectors with no duplicates and no orphans, every centroid is unit
    length (spherical k-means), and no shelf ended up dead-empty."""
    idx = build()
    filed = sorted(row for shelf in idx._lists for row in shelf)
    assert filed == list(range(N))
    assert all(abs(ivf.dot(c, c) - 1.0) < 1e-9 for c in idx._centroids)
    assert all(len(shelf) > 0 for shelf in idx._lists)


def test_ch6_probing_every_shelf_is_brute_force():
    """nprobe=nlist scans the whole library, so IVF degenerates into
    Chapter 2 exactly: identical top-10, recall 1.0 by construction.
    Approximation lives entirely in the shelves you skip."""
    idx = build()
    q = QUERIES[0]
    assert {i for _, i in idx.search(q, 10, nprobe=NLIST)} == true_top10(q)


def test_ch6_nprobe_is_the_recall_dial():
    """The chapter's core trade-off, measured: each extra probed shelf can
    only add candidates, so recall climbs with nprobe — and even 4 shelves
    of 32 (~12% of the library) should already find most of the truth."""
    idx = build()
    r1, r4, r16 = (recall_at_10(idx, p) for p in (1, 4, 16))
    assert r1 <= r4 <= r16
    assert r4 >= 0.60, f"recall@10 with nprobe=4 was only {r4:.2f}"
    assert r16 >= 0.95, f"recall@10 with nprobe=16 was only {r16:.2f}"


def test_ch6_ivf_scans_a_fraction_of_the_library():
    """The payoff arithmetic: a query costs ~nlist centroid checks plus the
    contents of nprobe shelves (~N·nprobe/nlist), far below N — and unlike
    HNSW there's no graph to walk, just flat lists to scan."""
    idx = build()
    idx.distance_evals = 0
    for q in QUERIES:
        idx.search(q, 10, nprobe=4)
    per_query = idx.distance_evals / len(QUERIES)
    expected = NLIST + N * 4 / NLIST            # ~220 for our shape
    assert per_query < N * 0.35, f"{per_query:.0f} evals vs {N} brute force"
    assert per_query < expected * 2             # and the formula holds


def test_ch6_delete_is_real_not_a_tombstone():
    """The structural advantage over Chapter 4: delete() removes the row
    from its shelf, so it can never be visited again — no tombstone set,
    no over-fetch, and the index does strictly LESS work afterwards."""
    idx = ivf.IVFIndex(dim=DIM, nlist=8)
    for v in DATASET[:200]:
        idx.add(v)
    idx.train()
    q = QUERIES[0]
    victim = idx.search(q, 1)[0][1]             # current best match
    idx.delete(victim)
    assert victim not in {i for _, i in idx.search(q, 10, nprobe=8)}
    assert sum(len(s) for s in idx._lists) == 199


def test_ch6_can_add_after_training():
    """Post-training adds are filed straight onto their nearest shelf and
    are immediately findable — a duplicate of an existing vector must come
    back as the top hit for itself."""
    idx = ivf.IVFIndex(dim=DIM, nlist=8)
    for v in DATASET[:200]:
        idx.add(v)
    idx.train()
    row = idx.add(DATASET[300])
    assert idx.search(DATASET[300], 1, nprobe=8)[0][1] == row


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed.")
