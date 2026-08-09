"""
test_vectordb.py — the guided tour.

Each test is a chapter claim from vectordb.py, restated as an assertion.
Read them in order; they build on each other. Runs with plain Python
(`python test_vectordb.py`) or pytest — no dependencies either way.
"""

import math
import os
import random
import tempfile

from vectordb import (BruteForceIndex, HNSWIndex, VectorDB, cosine, dot,
                      embed, normalize)

# ---------------------------------------------------------------------------
# A shared toy dataset: 800 random 16-dim vectors. Gaussian components make
# directions uniform on the sphere — the hardest, most honest case for ANN
# (real embeddings are easier: they live on low-dimensional manifolds).
# ---------------------------------------------------------------------------
N, DIM = 2000, 16
_rng = random.Random(42)
DATASET = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
QUERIES = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(20)]

_built = {}  # build each index once, share across tests


def build(kind):
    if kind not in _built:
        idx = {"flat": BruteForceIndex, "hnsw": HNSWIndex}[kind](DIM)
        for v in DATASET:
            idx.add(v)
        _built[kind] = idx
    return _built[kind]


# ---------------------------------------------------------------------------
# Chapter 1 — Similarity
# ---------------------------------------------------------------------------

def test_ch1_cosine_measures_direction_not_length():
    """cosine(a, b) only cares about the ANGLE between vectors: a vector is
    maximally similar to itself (1.0) and to any scaled copy of itself —
    which is exactly why magnitude is safe to throw away."""
    a = [1.0, 2.0, 3.0]
    assert math.isclose(cosine(a, a), 1.0)
    assert math.isclose(cosine(a, [10.0, 20.0, 30.0]), 1.0)   # same direction
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0)          # orthogonal
    assert math.isclose(cosine([1, 0], [-1, 0]), -1.0)        # opposite


def test_ch1_the_normalization_trick():
    """After normalizing both vectors, plain dot() equals cosine() — this
    identity is why every DB normalizes at insert time and never computes
    an actual cosine again."""
    a, b = [3.0, -1.0, 2.0], [0.5, 4.0, -2.0]
    assert math.isclose(dot(normalize(a), normalize(b)), cosine(a, b))


def test_ch1_zero_vector_is_rejected():
    """The zero vector has no direction, so cosine similarity to it is
    undefined (0/0). A good database refuses it loudly at the door instead
    of NaN-ing quietly at query time."""
    try:
        normalize([0.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Chapter 2 — Exact search
# ---------------------------------------------------------------------------

def test_ch2_brute_force_is_exact():
    """Brute force returns the TRUE nearest neighbors — we verify against an
    independent full sort. This is our ground truth for grading HNSW."""
    idx = build("flat")
    q = QUERIES[0]
    got = [i for _, i in idx.search(q, 10)]
    qn = normalize(q)
    want = sorted(range(N), key=lambda i: dot(qn, normalize(DATASET[i])),
                  reverse=True)[:10]
    assert got == want


def test_ch2_brute_force_cost_is_the_whole_dataset():
    """The problem statement, as an assertion: every query touches every
    vector. 800 today; 800 million at prod scale. This test motivates the
    entire next chapter."""
    idx = build("flat")
    before = idx.distance_evals
    idx.search(QUERIES[1], 10)
    assert idx.distance_evals - before == N


# ---------------------------------------------------------------------------
# Chapter 3 — HNSW
# ---------------------------------------------------------------------------

def recall_at_10(hnsw, flat, queries, ef=None):
    """Recall@10 = |HNSW's top-10 ∩ true top-10| / 10, averaged over queries.
    THE metric of the ANN world — every benchmark you've seen plots it."""
    total = 0.0
    for q in queries:
        truth = {i for _, i in flat.search(q, 10)}
        found = {i for _, i in hnsw.search(q, 10, ef=ef)}
        total += len(truth & found) / 10
    return total / len(queries)


def test_ch3_hnsw_finds_almost_everything():
    """'Approximate' is quantifiable, not hand-wavy: on uniformly random
    vectors (the worst case) HNSW at default settings should still find
    ≥90% of the true top-10."""
    r = recall_at_10(build("hnsw"), build("flat"), QUERIES)
    assert r >= 0.90, f"recall@10 = {r:.2f}"


def test_ch3_hnsw_does_a_fraction_of_the_work():
    """The payoff: HNSW answers queries while computing far fewer distances
    than the N=800 brute force needs. This gap WIDENS with N — brute force
    scales O(N), the graph walk ~O(log N)."""
    idx = build("hnsw")
    before = idx.distance_evals
    for q in QUERIES:
        idx.search(q, 10)
    per_query = (idx.distance_evals - before) / len(QUERIES)
    assert per_query < N * 0.6, f"{per_query:.0f} evals vs {N} brute force"


def test_ch3_ef_is_the_recall_speed_dial():
    """Crank ef down → faster but blinder. Crank it up → slower but nearly
    exact. This one knob is what 'tuning your vector DB' mostly means."""
    hnsw, flat = build("hnsw"), build("flat")
    assert recall_at_10(hnsw, flat, QUERIES, ef=200) >= \
           recall_at_10(hnsw, flat, QUERIES, ef=10) - 0.01  # noise margin


def test_ch3_the_graph_is_a_hierarchy():
    """The layer lottery worked: layer 0 holds every node, and each higher
    layer holds an exponentially thinner sample (the 'highway' shape)."""
    idx = build("hnsw")
    sizes = [sum(1 for nb in idx._neighbors if len(nb) > L)
             for L in range(idx._max_level + 1)]
    assert sizes[0] == N
    assert all(a > b for a, b in zip(sizes, sizes[1:]))


# ---------------------------------------------------------------------------
# Chapter 4 — The database shell
# ---------------------------------------------------------------------------

def make_db():
    db = VectorDB(dim=64)
    db.upsert("fern", embed("water your fern weekly"), {"topic": "plants"})
    db.upsert("cactus", embed("a cactus barely needs water"), {"topic": "plants"})
    db.upsert("git", embed("git reset undoes a commit"), {"topic": "code"})
    return db


def test_ch4_query_speaks_ids_and_metadata():
    """The shell translates: you never see internal row numbers, only your
    own ids and payloads, ranked by score."""
    hits = make_db().query(embed("how often to water a fern"), k=2)
    assert hits[0]["id"] == "fern"
    assert hits[0]["metadata"] == {"topic": "plants"}
    assert hits[0]["score"] >= hits[1]["score"]


def test_ch4_where_filters_but_search_still_ranks():
    """A filter changes WHO is eligible, not HOW they're ranked: asking a
    plant question with where={'topic': 'code'} must return only code docs."""
    hits = make_db().query(embed("how often to water a fern"), k=5,
                           where={"topic": "code"})
    assert [h["id"] for h in hits] == ["git"]


def test_ch4_delete_is_a_tombstone():
    """Deleted rows vanish from results instantly — but the vector is still
    inside the graph (len(_dead) grew). That gap between 'gone for you' and
    'gone for real' is the tombstone pattern."""
    db = make_db()
    db.delete("fern")
    assert "fern" not in [h["id"] for h in
                          db.query(embed("water a fern"), k=5)]
    assert len(db._dead) == 1


def test_ch4_upsert_replaces_and_bloats():
    """Re-upserting an id serves the NEW vector/metadata, while quietly
    tombstoning the old row — observable index bloat, the reason real DBs
    have an 'optimize' button."""
    db = make_db()
    db.upsert("fern", embed("ferns love humid bathrooms"), {"topic": "plants",
                                                            "v": 2})
    hits = db.query(embed("humid bathroom plant"), k=1)
    assert hits[0]["id"] == "fern" and hits[0]["metadata"]["v"] == 2
    assert len(db) == 3 and len(db._dead) == 1


# ---------------------------------------------------------------------------
# Chapter 5 — Persistence
# ---------------------------------------------------------------------------

def test_ch5_a_database_survives_a_restart():
    """Save, 'crash', load: the reborn DB gives byte-identical answers —
    same ids, same scores, same respect for tombstones."""
    db = make_db()
    db.delete("cactus")
    q = embed("watering plants")
    before = db.query(q, k=3)
    path = os.path.join(tempfile.mkdtemp(), "db.json")
    db.save(path)
    after = VectorDB.load(path).query(q, k=3)
    assert before == after
    assert "cactus" not in [h["id"] for h in after]


# ---------------------------------------------------------------------------
# Plain-python runner (pytest also picks these up automatically)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed.")
