"""
vectordb.py — a vector database from scratch, in one file you can read in an hour.

You use a vector database every day. Every RAG pipeline, every "chat with your
docs" product, every semantic-search box is one. And yet if someone asks
"what does Pinecone/Qdrant/pgvector *actually do*?", most engineers can only
say "uh... it finds similar vectors, fast". This file is the missing answer.

It is a real, working vector database:

    db = VectorDB(dim=64)
    db.upsert("doc-1", embed("how to water a fern"), {"topic": "plants"})
    db.query(embed("keeping plants alive"), k=3, where={"topic": "plants"})

with the four things that make a vector DB a vector DB:

    Chapter 1 — Similarity.      What "close" means for vectors, and the
                                 normalization trick every real DB uses.
    Chapter 2 — Exact search.    Brute force: the baseline that is 100%
                                 correct and 100% too slow at scale.
    Chapter 3 — HNSW.            The approximate-nearest-neighbor graph index
                                 inside Qdrant, Weaviate, pgvector, Milvus...
                                 ~150 lines, and you'll finally get it.
    Chapter 4 — The shell.       String IDs, metadata, filters, deletes —
                                 the "database" wrapped around the index.
    Chapter 5 — Persistence.     Saving the whole thing to disk and back.

Rules of the house: pure Python standard library. No numpy, no C extensions.
It's 100-1000x slower than faiss — on purpose. Speed hides ideas; this file
is the ideas. Run `python vectordb.py` for a demo, and read
test_vectordb.py as a guided tour with assertions.
"""

from __future__ import annotations

import heapq
import json
import math
import random
import zlib

# A vector is just a list of floats. Real DBs use packed float32 arrays
# (4 bytes/dim, SIMD-friendly); a Python list of boxed floats is ~8x fatter.
# Same idea, worse constant factor.
Vector = list[float]


# ---------------------------------------------------------------------------
# Chapter 1 — Similarity: what does "close" mean?
# ---------------------------------------------------------------------------
# Embedding models turn text/images into points in R^d such that *semantic*
# similarity becomes *geometric* closeness. A vector DB's whole job is:
# "given query point q, which stored points are closest?"
#
# The metric almost everyone uses is cosine similarity — the angle between
# vectors, ignoring their length. Why ignore length? Because embedding
# magnitude mostly encodes junk (text length, token frequency quirks), while
# *direction* encodes meaning.


def dot(a: Vector, b: Vector) -> float:
    """Σ aᵢ·bᵢ — the inner loop of every vector database on earth.

    When faiss or pgvector burn CPU, they are burning it here, vectorized
    with AVX-512 across 8-16 floats per instruction. We do it one float at
    a time, which is the same computation in slow motion.
    """
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """Euclidean length ‖a‖ = sqrt(Σ aᵢ²)."""
    return math.sqrt(dot(a, a))


def cosine(a: Vector, b: Vector) -> float:
    """cos(θ) = a·b / (‖a‖·‖b‖) — in [-1, 1], where 1 means 'same direction'."""
    return dot(a, b) / (norm(a) * norm(b))


def normalize(a: Vector) -> Vector:
    """Scale a vector to length 1.

    THE trick of the trade: if you normalize every vector once at insert
    time, then cosine(a, b) == dot(a, b) — two multiplies per dimension
    instead of two square roots and a division per comparison. Every real
    vector DB does this when you ask for cosine distance. So do we: from
    here on, everything stored is unit-length and 'similarity' means dot().
    """
    n = norm(a)
    if n == 0:
        raise ValueError("cannot index the zero vector (it has no direction)")
    return [x / n for x in a]


# ---------------------------------------------------------------------------
# Chapter 2 — Exact search: brute force, the honest baseline
# ---------------------------------------------------------------------------
# The simplest possible index: keep a list, compare the query with EVERYTHING,
# return the top k. It is perfectly accurate. It is also O(N·d) per query:
# at 10M vectors × 1536 dims that's ~15 billion multiplies per query.
# Everything else in this file exists because of this paragraph.


class BruteForceIndex:
    """Exact nearest-neighbor search by exhaustive scan.

    Not a strawman! Below ~50k vectors this is often the RIGHT choice:
    zero build cost, zero recall loss, trivially correct. faiss calls it
    IndexFlat and it's the recommended starting point.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: list[Vector] = []   # row i = vector with internal id i
        self.distance_evals = 0            # bookkeeping so tests can *prove*
                                           # HNSW does less work (Chapter 3)

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector: Vector) -> int:
        """Store a vector, return its internal integer id (its row number)."""
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        self._vectors.append(normalize(vector))
        return len(self._vectors) - 1

    def search(self, query: Vector, k: int) -> list[tuple[float, int]]:
        """Return the k best (similarity, internal_id) pairs, best first."""
        q = normalize(query)
        self.distance_evals += len(self._vectors)
        scored = ((dot(q, v), i) for i, v in enumerate(self._vectors))
        # heapq.nlargest is an O(N log k) top-k — better than sorting all N.
        return heapq.nlargest(k, scored)

    # (Persistence for Chapter 5 — just the raw rows.)
    def to_dict(self) -> dict:
        return {"dim": self.dim, "vectors": self._vectors}

    @classmethod
    def from_dict(cls, d: dict) -> "BruteForceIndex":
        idx = cls(d["dim"])
        idx._vectors = d["vectors"]        # already normalized when stored
        return idx


# ---------------------------------------------------------------------------
# Chapter 3 — HNSW: the index you use daily but can't explain
# ---------------------------------------------------------------------------
# Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016) are THE
# workhorse ANN index: Qdrant, Weaviate, Milvus, pgvector, Vespa, Redis —
# all HNSW. The idea fits in three sentences:
#
#   1. Connect each vector to its ~M nearest neighbors → a graph you can
#      "walk" toward any query by always stepping to the neighbor closest
#      to it (greedy search).
#   2. A flat graph makes greedy walks start slow (many tiny steps from a
#      random start) and get stuck in local optima. So build LAYERS, like
#      highway systems: layer 2 has ~1% of nodes (long hops), layer 1 has
#      ~10%, layer 0 has everyone. Start at the top, take big hops, drop
#      down a layer whenever you can't improve. Zoom out, then zoom in.
#   3. To make it robust, don't walk with a single "current best" but with
#      a beam of `ef` candidates. Bigger ef = better recall, more work.
#
# That's the entire trick. Search becomes ~O(log N) hops instead of O(N)
# comparisons, at the price of being *approximate*: it can miss a true
# neighbor. Tests in Chapter 3 of test_vectordb.py measure exactly how often.


class HNSWIndex:
    """A compact-but-faithful HNSW. Same API as BruteForceIndex."""

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 100,
                 ef_search: int = 64, seed: int = 0):
        self.dim = dim
        self._m = m                    # max links per node on layers ≥ 1
        self._m0 = 2 * m               # layer 0 is denser (paper's default)
        self._ef_construction = ef_construction  # beam width while BUILDING
        self._ef_search = ef_search              # beam width while QUERYING
        # Each node's top layer is drawn from a geometric-ish distribution:
        # P(level ≥ L) = (1/M)^L. With M=16, ~94% of nodes live only on
        # layer 0, ~6% reach layer 1, ~0.4% reach layer 2... exactly the
        # "few long-haul airports, many local roads" shape we want.
        self._level_mult = 1.0 / math.log(m)
        self._rng = random.Random(seed)  # seeded → identical graph every run
        self._vectors: list[Vector] = []
        # _neighbors[i][L] = list of node ids linked to node i on layer L.
        self._neighbors: list[list[list[int]]] = []
        self._entry: int | None = None   # id of the node where searches start
        self._max_level = 0              # highest occupied layer
        self.distance_evals = 0

    def __len__(self) -> int:
        return len(self._vectors)

    def _dot(self, q: Vector, node: int) -> float:
        """All similarity flows through here so distance_evals stays honest."""
        self.distance_evals += 1
        return dot(q, self._vectors[node])

    def _random_level(self) -> int:
        # -ln(U) is an exponential random variable; scaling by 1/ln(M) and
        # flooring gives the geometric layer distribution described above.
        return int(-math.log(self._rng.random()) * self._level_mult)

    # -- the one algorithm that matters: a greedy beam search on one layer --
    def _search_layer(self, q: Vector, entry: list[int], ef: int,
                      layer: int) -> list[tuple[float, int]]:
        """From `entry` nodes, walk layer `layer` toward q with beam width ef.

        Two heaps, one invariant:
          * `candidates`: frontier still to expand, best-similarity first.
          * `results`:    the ef best nodes seen so far (a min-heap, so the
                          WORST of the best sits at results[0], cheap to evict).
        Stop when the best unexplored candidate is worse than the worst
        result — the frontier can no longer improve the answer. This early
        exit is why HNSW is fast; everything else is bookkeeping.
        """
        sims = {e: self._dot(q, e) for e in entry}
        visited = set(entry)
        candidates = [(-s, e) for e, s in sims.items()]  # max-heap via minus
        heapq.heapify(candidates)
        results = [(s, e) for e, s in sims.items()]      # min-heap, natural
        heapq.heapify(results)
        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break                        # frontier can't beat our worst
            for nb in self._neighbors[node][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._dot(q, nb)
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(results, (s, nb))
                    if len(results) > ef:
                        heapq.heappop(results)   # evict current worst
        return sorted(results, reverse=True)     # best first

    def add(self, vector: Vector) -> int:
        """Insert = search for where the vector belongs, then link it in."""
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        q = normalize(vector)
        new = len(self._vectors)
        self._vectors.append(q)
        level = self._random_level()
        self._neighbors.append([[] for _ in range(level + 1)])

        if self._entry is None:              # very first node: nothing to link
            self._entry, self._max_level = new, level
            return new

        # Phase 1: from the top of the graph down to just above the new
        # node's level, greedy-descend with a beam of 1 (we only need a
        # good entry point, not a full result set).
        ep = self._entry
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]

        # Phase 2: on every layer the new node occupies, find its true
        # neighborhood with the wide construction beam, then wire it in.
        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(q, [ep], self._ef_construction, layer)
            cap = self._m0 if layer == 0 else self._m
            selected = [node for _, node in found[: self._m]]
            self._neighbors[new][layer] = selected
            for nb in selected:
                links = self._neighbors[nb][layer]
                links.append(new)             # links are bidirectional
                if len(links) > cap:
                    # Neighbor over capacity → keep only its `cap` closest
                    # links. (The paper has a smarter diversity heuristic
                    # here — see the "extra chapters wanted" list.)
                    v = self._vectors[nb]
                    links.sort(key=lambda o: self._dot(v, o), reverse=True)
                    del links[cap:]
            ep = found[0][1]                  # best match seeds next layer

        if level > self._max_level:           # new node is the tallest →
            self._entry = new                 # it becomes the global entry
            self._max_level = level
        return new

    def search(self, query: Vector, k: int,
               ef: int | None = None) -> list[tuple[float, int]]:
        """Top-k (similarity, internal_id): descend the highway layers with
        beam 1, then sweep layer 0 with the full beam. ef is your live
        recall-vs-speed dial — the same knob you tune in production DBs."""
        if self._entry is None:
            return []
        ef = max(ef or self._ef_search, k)
        q = normalize(query)
        ep = self._entry
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]
        return self._search_layer(q, [ep], ef, 0)[:k]

    # -- persistence (Chapter 5): the whole index is 5 plain fields --------
    def to_dict(self) -> dict:
        return {"dim": self.dim, "m": self._m,
                "ef_construction": self._ef_construction,
                "ef_search": self._ef_search, "entry": self._entry,
                "max_level": self._max_level, "vectors": self._vectors,
                "neighbors": self._neighbors}

    @classmethod
    def from_dict(cls, d: dict) -> "HNSWIndex":
        idx = cls(d["dim"], m=d["m"], ef_construction=d["ef_construction"],
                  ef_search=d["ef_search"])
        idx._vectors = d["vectors"]
        idx._neighbors = d["neighbors"]
        idx._entry = d["entry"]
        idx._max_level = d["max_level"]
        return idx


# ---------------------------------------------------------------------------
# Chapter 4 — The database shell: what makes an index a database
# ---------------------------------------------------------------------------
# An index maps vectors to integers. A DATABASE speaks your language:
# string IDs, metadata payloads, filters, upserts, deletes. This wrapper is
# unglamorous — and it's most of the code in real vector DBs too.
#
# The interesting wart: HNSW CANNOT DELETE. Removing a node would tear holes
# in the graph that greedy search falls into. So, like Qdrant and friends,
# we use TOMBSTONES: mark the row dead, skip it in results, and over-fetch
# to compensate. (Real DBs eventually rebuild the graph to purge tombstones
# — that's what "vacuum"/"optimize" does.)


class VectorDB:
    """The user-facing database: upsert / query / delete / save / load."""

    _INDEXES = {"hnsw": HNSWIndex, "flat": BruteForceIndex}

    def __init__(self, dim: int, index: str = "hnsw", **index_params):
        self.dim = dim
        self._index_type = index
        self._index = self._INDEXES[index](dim, **index_params)
        self._row_of: dict[str, int] = {}   # user id  -> internal row
        self._id_of: dict[int, str] = {}    # internal row -> user id
        self._meta: dict[str, dict] = {}    # user id  -> metadata payload
        self._dead: set[int] = set()        # tombstoned internal rows

    def __len__(self) -> int:
        return len(self._row_of)

    def upsert(self, id: str, vector: Vector, metadata: dict | None = None):
        """Insert or replace. 'Replace' is a lie every HNSW-backed DB tells:
        the old row is tombstoned and the vector is inserted as a NEW node,
        because the graph can't rewire in place. Watch len(self._dead) grow
        if you upsert the same id in a loop — that's real 'index bloat'."""
        if id in self._row_of:
            self._dead.add(self._row_of[id])
        row = self._index.add(vector)
        self._row_of[id] = row
        self._id_of[row] = id
        self._meta[id] = metadata or {}

    def delete(self, id: str):
        """Tombstone only — O(1), no graph surgery. See chapter intro."""
        self._dead.add(self._row_of.pop(id))
        self._meta.pop(id)

    def query(self, vector: Vector, k: int = 5,
              where: dict | None = None) -> list[dict]:
        """Top-k live rows as {id, score, metadata}, filtered by `where`
        (exact-match on metadata keys, the classic 80% filter).

        This is POST-filtering: search first, discard mismatches after. To
        survive discards we over-fetch — k extra results per tombstone plus
        a fudge factor. It's the honest hack most DBs started with; its
        failure mode (a selective filter needs a huge over-fetch) is exactly
        why 'filtered ANN' is still an active research area. Qdrant's
        in-graph filtering chapter would slot in right here.
        """
        fetch = min((k + len(self._dead)) * 2 + 10, len(self._index))
        out = []
        for score, row in self._index.search(vector, fetch):
            if row in self._dead:
                continue
            id = self._id_of[row]
            md = self._meta[id]
            if where and any(md.get(key) != val for key, val in where.items()):
                continue
            out.append({"id": id, "score": score, "metadata": md})
            if len(out) == k:
                break
        return out

    # -----------------------------------------------------------------------
    # Chapter 5 — Persistence: a database survives a restart
    # -----------------------------------------------------------------------
    # We serialize EVERYTHING — vectors, graph links, id maps, tombstones —
    # to one JSON file. JSON so you can `cat` your database and see there's
    # no magic inside. Real engines differ in degree, not kind: binary
    # layouts, mmap so the OS pages the file in lazily, and a write-ahead
    # log so a crash mid-write can't corrupt the file. Same five fields.

    def save(self, path: str):
        state = {"dim": self.dim, "index_type": self._index_type,
                 "index": self._index.to_dict(), "row_of": self._row_of,
                 "meta": self._meta, "dead": sorted(self._dead)}
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "VectorDB":
        with open(path) as f:
            state = json.load(f)
        db = cls.__new__(cls)                # skip __init__; restore fields
        db.dim = state["dim"]
        db._index_type = state["index_type"]
        db._index = cls._INDEXES[db._index_type].from_dict(state["index"])
        db._row_of = state["row_of"]
        db._id_of = {row: id for id, row in db._row_of.items()}
        db._meta = state["meta"]
        db._dead = set(state["dead"])
        return db


# ---------------------------------------------------------------------------
# Epilogue — a demo with zero dependencies (yes, including the embeddings)
# ---------------------------------------------------------------------------
# Real embeddings come from a neural network. But ANY function text→vector
# where similar texts land near each other will demo the machinery. Hashed
# character trigrams are the world's worst embedding model — and completely
# transparent, which tonight is the better trade.


def embed(text: str, dim: int = 64) -> Vector:
    """Bag of character trigrams, feature-hashed into `dim` buckets."""
    v = [0.0] * dim
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = zlib.crc32(t[i:i + 3].encode())     # crc32: stable across runs,
        v[h % dim] += 1.0 if h & 1 else -1.0    # unlike Python's hash()
    return v


if __name__ == "__main__":
    docs = {
        "fern-care": ("Water your fern twice a week and keep it in "
                      "indirect sunlight", {"topic": "plants"}),
        "git-undo": ("To undo the last git commit, use git reset "
                     "with the --soft flag", {"topic": "programming"}),
        "pho-recipe": ("Simmer beef bones for hours to make a rich "
                       "phở broth", {"topic": "cooking"}),
        "cactus-care": ("A cactus needs very little water and lots of "
                        "direct sun", {"topic": "plants"}),
    }
    db = VectorDB(dim=64)
    for id, (text, meta) in docs.items():
        db.upsert(id, embed(text), meta)

    print("query: 'how do I keep my plant alive?'\n")
    for hit in db.query(embed("how do I keep my plant alive?"), k=3):
        print(f"  {hit['score']:+.3f}  {hit['id']:<12} {hit['metadata']}")

    print("\nsame query, where={'topic': 'programming'}:\n")
    for hit in db.query(embed("how do I keep my plant alive?"), k=3,
                        where={"topic": "programming"}):
        print(f"  {hit['score']:+.3f}  {hit['id']:<12} {hit['metadata']}")
