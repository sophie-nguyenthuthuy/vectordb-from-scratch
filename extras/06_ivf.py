"""
Extra chapter 6 — IVF: the OTHER big ANN family.

Chapter 3 solved "too many comparisons" with a graph you walk. IVF
(Inverted File index — faiss's IndexIVFFlat, and the backbone of IVFPQ,
which powers a lot of billion-scale search) solves it with a map you
consult first:

    1. TRAIN:  cluster all vectors into `nlist` cells with k-means.
    2. INDEX:  file each vector under its nearest centroid — an
               "inverted list" per cell, like shelves in a library.
    3. SEARCH: compare the query with the nlist centroids only, walk to
               the best `nprobe` shelves, and scan just those.

Cost per query drops from N to about nlist + N·(nprobe/nlist) — probe 4 of
64 shelves and you scan ~6% of the library. The failure mode is the same
picture: a true neighbor filed on a shelf you didn't probe is invisible.
`nprobe` is IVF's recall dial, exactly as `ef` was HNSW's.

Why choose IVF over HNSW? Three classic reasons: memory (lists are flat
arrays, no graph pointers), quantization (each shelf compresses well —
that's IVFPQ), and DELETES — pull a vector off its shelf and it is gone,
no tombstones (contrast with Chapter 4!). The price: k-means needs a
representative TRAINING step up front, and recall at the same speed is
usually a notch below a well-tuned HNSW.

This chapter is self-contained on purpose (stands alone, runs alone); the
two primitives below are re-derived from Chapter 1.
"""

import heapq
import math
import random

Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    """Chapter 1's inner loop, again — it is always this loop."""
    return sum(x * y for x, y in zip(a, b))


def normalize(a: Vector) -> Vector:
    """Unit length, so cosine == dot (the Chapter 1 trick)."""
    n = math.sqrt(dot(a, a))
    if n == 0:
        raise ValueError("cannot index the zero vector (it has no direction)")
    return [x / n for x in a]


def kmeans(vectors: list[Vector], k: int, iters: int = 8,
           seed: int = 0) -> tuple[list[Vector], list[list[int]]]:
    """Lloyd's k-means, the 1957 algorithm that still trains every IVF index.

    Two steps, repeated: ASSIGN each point to its nearest centroid, then
    UPDATE each centroid to the mean of its points. Each step can only
    improve the fit, so it converges (to a local optimum — good enough).

    One wrinkle because we live on the unit sphere (Chapter 1): after
    averaging we re-normalize the centroid. That makes this *spherical*
    k-means — the variant that matches cosine similarity.

    Returns (centroids, clusters) where clusters[c] lists the indices of
    the vectors assigned to centroid c.
    """
    rng = random.Random(seed)
    # Seed centroids with k distinct real points (k-means++ would pick
    # smarter seeds; random works fine at chapter scale).
    centroids = [vectors[i][:] for i in rng.sample(range(len(vectors)), k)]
    clusters: list[list[int]] = [[] for _ in range(k)]
    for _ in range(iters):
        clusters = [[] for _ in range(k)]
        for vi, v in enumerate(vectors):                      # ASSIGN
            best = max(range(k), key=lambda c: dot(v, centroids[c]))
            clusters[best].append(vi)
        for c in range(k):                                    # UPDATE
            if not clusters[c]:
                # An empty cell is a dead shelf. Standard fix (faiss does a
                # version of this): re-seed it on a random point.
                centroids[c] = vectors[rng.randrange(len(vectors))][:]
                continue
            dim = len(vectors[0])
            mean = [sum(vectors[i][d] for i in clusters[c]) / len(clusters[c])
                    for d in range(dim)]
            centroids[c] = normalize(mean)
    return centroids, clusters


class IVFIndex:
    """Same API as Chapter 2/3 indexes — add / search / distance_evals —
    plus the two things HNSW can't offer: train() and a real delete()."""

    def __init__(self, dim: int, nlist: int = 32, nprobe: int = 4,
                 seed: int = 0):
        self.dim = dim
        self._nlist = nlist        # number of cells ("shelves")
        self._nprobe = nprobe      # shelves visited per query: the dial
        self._seed = seed
        self._vectors: list[Vector] = []
        self._centroids: list[Vector] | None = None   # None = not trained
        self._lists: list[list[int]] = []             # cell -> vector rows
        self._cell_of: list[int] = []                 # row -> its cell
        self.distance_evals = 0

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector: Vector) -> int:
        """Before training: just collect. After training: file the new
        vector straight onto its nearest shelf. (Real systems do exactly
        this — and if the data distribution drifts far from what k-means
        saw, the shelves stop matching reality and you retrain. 'My IVF
        recall decayed over months' is this sentence in production.)"""
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        v = normalize(vector)
        self._vectors.append(v)
        row = len(self._vectors) - 1
        if self._centroids is not None:
            c = self._nearest_cell(v)
            self._lists[c].append(row)
            self._cell_of.append(c)
        return row

    def _nearest_cell(self, v: Vector) -> int:
        self.distance_evals += self._nlist
        return max(range(self._nlist),
                   key=lambda c: dot(v, self._centroids[c]))

    def train(self):
        """Run k-means over everything indexed so far and build the shelves.
        This up-front, whole-dataset step is THE structural difference from
        HNSW, which digests one vector at a time forever."""
        self._centroids, self._lists = kmeans(self._vectors, self._nlist,
                                              seed=self._seed)
        self._cell_of = [0] * len(self._vectors)
        for c, rows in enumerate(self._lists):
            for row in rows:
                self._cell_of[row] = c

    def delete(self, row: int):
        """The IVF party trick: pull the vector off its shelf and no probe
        path can ever reach it again. O(shelf length), no tombstones, no
        graph surgery — the delete HNSW users dream about (Chapter 4)."""
        self._lists[self._cell_of[row]].remove(row)

    def search(self, query: Vector, k: int,
               nprobe: int | None = None) -> list[tuple[float, int]]:
        """Rank shelves by centroid similarity, exhaustively scan the best
        `nprobe` of them, top-k the union. Auto-trains on first use."""
        if self._centroids is None:
            self.train()
        nprobe = min(nprobe or self._nprobe, self._nlist)
        q = normalize(query)
        self.distance_evals += self._nlist
        ranked = heapq.nlargest(nprobe, range(self._nlist),
                                key=lambda c: dot(q, self._centroids[c]))
        scored = []
        for c in ranked:
            self.distance_evals += len(self._lists[c])
            scored += [(dot(q, self._vectors[row]), row)
                       for row in self._lists[c]]
        return heapq.nlargest(k, scored)


if __name__ == "__main__":
    # The whole chapter in one table: turn the nprobe dial, watch recall
    # buy itself with distance computations. Brute force = 2000 evals/query.
    rng = random.Random(7)
    data = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(2000)]
    queries = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(20)]
    truth = []
    for q in queries:
        qn = normalize(q)
        truth.append({i for _, i in heapq.nlargest(
            10, ((dot(qn, normalize(v)), i) for i, v in enumerate(data)))})

    idx = IVFIndex(dim=16, nlist=32)
    for v in data:
        idx.add(v)
    idx.train()

    print("nprobe   recall@10   distance evals/query   (brute force: 2000)")
    for nprobe in (1, 2, 4, 8, 16, 32):
        idx.distance_evals = 0
        recall = sum(
            len(truth[qi] & {i for _, i in idx.search(q, 10, nprobe=nprobe)})
            / 10 for qi, q in enumerate(queries)) / len(queries)
        print(f"{nprobe:>6}   {recall:>9.2f}   {idx.distance_evals // len(queries):>20}")
