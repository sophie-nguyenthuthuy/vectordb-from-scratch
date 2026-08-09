"""
Extra chapter 7 — Product Quantization: the memory story.

Chapters 3 and 6 attacked TIME (fewer comparisons per query). PQ attacks
SPACE. Do the arithmetic that ruins RAG budgets: 10M vectors × 1536 dims
× 4 bytes = 61 GB of RAM — for ONE index. PQ (Jégou, Douze & Schmid,
2011) shrinks vectors 16-64× and, remarkably, still searches them
WITHOUT decompressing. It's the "PQ" in faiss's IVFPQ, the format behind
most billion-scale deployments.

Three ideas, stacked:

    1. SPLIT.    Chop each d-dim vector into `m` subvectors of d/m dims.
                 A 16-dim vector becomes 4 chunks of 4 dims.
    2. QUANTIZE. Per chunk position, k-means the whole dataset's chunks
                 into `k` "codewords" (a codebook). Now any chunk can be
                 replaced by the ID of its nearest codeword: a number in
                 [0, k). The vector of m chunk-IDs is the CODE — with
                 k=256 that's ONE BYTE per chunk. 1536 floats → 8-96 bytes.
                 (Why per-chunk codebooks instead of one k-means over whole
                 vectors? Combinatorics: m codebooks of k entries give
                 k^m distinct reconstructions — 256^8 ≈ 10^19 regions from
                 only 256·8 stored centroids. That product is the "Product"
                 in the name.)
    3. ADC.      To search, DON'T decompress. Precompute, once per query,
                 table[j][c] = dot(query chunk j, codeword c of book j) —
                 m·k tiny dot products. Then any stored vector's score is
                 just m table lookups + adds: sum(table[j][code[j]]).
                 That's Asymmetric Distance Computation: exact query vs
                 quantized database, floats vs bytes.

The price is that scores are now APPROXIMATE (quantization error), so the
ranking wobbles. The standard cure ships in every serious engine: use the
cheap PQ scan to shortlist `rerank` candidates, then rescore just those
few with full-precision vectors fetched from slow storage. RAM holds
bytes; disk holds truth.

Self-contained like every extra chapter; dot() is Chapter 1's, and this
k-means is the PLAIN Euclidean one — compare Chapter 6, which needed the
spherical variant. Chunks of a unit vector are not unit vectors, so here
we minimize reconstruction error, not angle.
"""

import heapq
import math
import random

Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    """Chapter 1's inner loop. In PQ it survives only in two cheap places:
    building the query's lookup tables, and the optional rerank."""
    return sum(x * y for x, y in zip(a, b))


def normalize(a: Vector) -> Vector:
    """Unit length so cosine == dot, as always (Chapter 1)."""
    n = math.sqrt(dot(a, a))
    if n == 0:
        raise ValueError("cannot index the zero vector (it has no direction)")
    return [x / n for x in a]


def kmeans(points: list[Vector], k: int, iters: int = 8,
           seed: int = 0) -> list[Vector]:
    """Plain Euclidean k-means: assign to nearest centroid by squared
    distance, update to the raw mean. No sphere games here — a codebook's
    job is to RECONSTRUCT chunks with minimal error, and the mean is the
    point that minimizes summed squared error to its cluster."""
    rng = random.Random(seed)
    dim = len(points[0])
    centroids = [points[i][:] for i in rng.sample(range(len(points)), k)]

    def sqdist(p, c):
        return sum((x - y) ** 2 for x, y in zip(p, c))

    for _ in range(iters):
        clusters = [[] for _ in range(k)]
        for p in points:
            clusters[min(range(k),
                         key=lambda c: sqdist(p, centroids[c]))].append(p)
        for c in range(k):
            if not clusters[c]:               # dead codeword → reseed it
                centroids[c] = points[rng.randrange(len(points))][:]
                continue
            centroids[c] = [sum(p[d] for p in clusters[c]) / len(clusters[c])
                            for d in range(dim)]
    return centroids


class PQIndex:
    """Vectors in, byte-codes stored, search by table lookups.

    Keeps the originals too — but ONLY the codes are 'in RAM' in the cost
    model. Real systems put originals on disk (or drop them); we keep them
    because the rerank step needs somewhere to fetch truth from.
    """

    def __init__(self, dim: int, m: int = 4, k: int = 32, seed: int = 0):
        if dim % m:
            raise ValueError(f"dim={dim} must divide into m={m} chunks")
        self.dim = dim
        self._m = m                  # chunks per vector
        self._sub = dim // m         # dims per chunk
        self._k = k                  # codewords per codebook (256 → 1 byte)
        self._seed = seed
        self._books: list[list[Vector]] | None = None   # m codebooks
        self._codes: list[list[int]] = []               # the compressed DB
        self._originals: list[Vector] = []              # "disk", for rerank

    def __len__(self) -> int:
        return len(self._codes)

    def _chunks(self, v: Vector) -> list[Vector]:
        return [v[j * self._sub:(j + 1) * self._sub] for j in range(self._m)]

    def add(self, vector: Vector) -> int:
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        v = normalize(vector)
        self._originals.append(v)
        if self._books is not None:
            self._codes.append(self._encode(v))
        return len(self._originals) - 1

    def train(self):
        """One k-means per chunk position, then compress everything.
        Chunk j of every vector trains codebook j — position matters,
        because embedding dimensions are not interchangeable."""
        cols = [[self._chunks(v)[j] for v in self._originals]
                for j in range(self._m)]
        self._books = [kmeans(cols[j], self._k, seed=self._seed + j)
                       for j in range(self._m)]
        self._codes = [self._encode(v) for v in self._originals]

    def _encode(self, v: Vector) -> list[int]:
        """Vector → m small integers: each chunk becomes the index of its
        nearest codeword. This line is where the compression happens."""
        return [min(range(self._k),
                    key=lambda c: sum((x - y) ** 2 for x, y in
                                      zip(chunk, self._books[j][c])))
                for j, chunk in enumerate(self._chunks(v))]

    def decode(self, row: int) -> Vector:
        """Code → approximate vector: concatenate its codewords. Search
        never calls this (that's the point of ADC) — it exists so tests
        can measure exactly what the compression threw away."""
        return [x for j, c in enumerate(self._codes[row])
                for x in self._books[j][c]]

    def bytes_per_vector(self) -> tuple[int, int]:
        """(uncompressed, compressed) sizes in the standard cost model:
        float32 originals vs ceil(log2(k)/8) bytes per chunk code."""
        return 4 * self.dim, self._m * max(1, math.ceil(
            math.log2(self._k) / 8))

    def search(self, query: Vector, topk: int,
               rerank: int = 0) -> list[tuple[float, int]]:
        """ADC scan; optionally rescore the best `rerank` codes exactly.

        Note what the hot loop does NOT contain: a single multiplication.
        All the float math collapsed into the m×k table up front; each of
        the N stored vectors costs m integer lookups and adds. This shape
        (tiny dense table + byte codes) is also why real PQ scans SIMD so
        absurdly well.
        """
        if self._books is None:
            self.train()
        q = normalize(query)
        table = [[dot(chunk, w) for w in self._books[j]]
                 for j, chunk in enumerate(self._chunks(q))]
        scored = ((sum(table[j][cj] for j, cj in enumerate(code)), row)
                  for row, code in enumerate(self._codes))
        if not rerank:
            return heapq.nlargest(topk, scored)
        # The two-tier trick: bytes vote for a shortlist, floats decide.
        shortlist = heapq.nlargest(max(rerank, topk), scored)
        return heapq.nlargest(topk, ((dot(q, self._originals[row]), row)
                                     for _, row in shortlist))


if __name__ == "__main__":
    # The chapter in two tables: what compression costs, what rerank buys.
    rng = random.Random(7)
    data = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(2000)]
    queries = [[rng.gauss(0, 1) for _ in range(16)] for _ in range(20)]
    truth = []
    for q in queries:
        qn = normalize(q)
        truth.append({i for _, i in heapq.nlargest(
            10, ((dot(qn, normalize(v)), i) for i, v in enumerate(data)))})

    idx = PQIndex(dim=16, m=4, k=32)
    for v in data:
        idx.add(v)
    idx.train()
    raw, packed = idx.bytes_per_vector()
    print(f"memory: {raw} bytes/vector -> {packed} bytes/vector "
          f"({raw // packed}x smaller)\n")
    # Note the rerank=10 row: rescoring exactly topk candidates reorders
    # them but can't recover anything the bytes missed — rerank only helps
    # when the shortlist is DEEPER than what you return.
    print("rerank   recall@10   (0 = trust the bytes alone)")
    for rerank in (0, 10, 50, 100, 200):
        recall = sum(
            len(truth[qi] & {i for _, i in idx.search(q, 10, rerank=rerank)})
            / 10 for qi, q in enumerate(queries)) / len(queries)
        print(f"{rerank:>6}   {recall:>9.2f}")
