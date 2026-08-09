"""
vectordb.py — một vector database viết từ đầu, trong một file bạn đọc hết trong một giờ.

Bạn dùng vector database mỗi ngày. Mọi pipeline RAG, mọi sản phẩm "chat với
tài liệu", mọi ô tìm kiếm ngữ nghĩa đều là một cái. Vậy mà nếu ai đó hỏi
"Pinecone/Qdrant/pgvector *thực sự làm gì*?", đa số kỹ sư chỉ trả lời được
"ờ... nó tìm các vector giống nhau, nhanh". File này là câu trả lời còn thiếu.

Đây là một vector database thật sự, chạy được:

    db = VectorDB(dim=64)
    db.upsert("doc-1", embed("how to water a fern"), {"topic": "plants"})
    db.query(embed("keeping plants alive"), k=3, where={"topic": "plants"})

với đủ bốn thứ làm nên một vector DB đúng nghĩa:

    Chương 1 — Độ tương đồng.  "Gần nhau" nghĩa là gì với vector, và mẹo
                               normalize mà mọi DB thật đều dùng.
    Chương 2 — Tìm chính xác.  Brute force: baseline đúng 100% và cũng
                               chậm 100% khi dữ liệu lớn.
    Chương 3 — HNSW.           Chỉ mục approximate-nearest-neighbor dạng đồ
                               thị bên trong Qdrant, Weaviate, pgvector,
                               Milvus... ~150 dòng, và bạn sẽ hiểu nó thật sự.
    Chương 4 — Lớp vỏ.         ID dạng chuỗi, metadata, filter, xoá —
                               phần "database" bọc quanh chỉ mục.
    Chương 5 — Lưu trữ.        Ghi toàn bộ xuống đĩa và đọc lại.

Luật chơi: chỉ dùng thư viện chuẩn của Python. Không numpy, không C
extension. Nó chậm hơn faiss 100-1000 lần — một cách cố ý. Tốc độ che mất
ý tưởng; file này là ý tưởng. Chạy `python vectordb.py` để xem demo, và đọc
test_vectordb.py như một tour có hướng dẫn kèm assertion.
"""

from __future__ import annotations

import heapq
import json
import math
import random
import zlib

# Một vector chỉ là một list số thực. DB thật dùng mảng float32 đóng gói
# (4 byte/chiều, thân thiện SIMD); list Python chứa float boxed nặng gấp ~8
# lần. Cùng một ý tưởng, hằng số tệ hơn.
Vector = list[float]


# ---------------------------------------------------------------------------
# Chương 1 — Độ tương đồng: "gần nhau" nghĩa là gì?
# ---------------------------------------------------------------------------
# Model embedding biến văn bản/ảnh thành các điểm trong R^d sao cho sự tương
# đồng *ngữ nghĩa* trở thành sự gần gũi *hình học*. Toàn bộ công việc của một
# vector DB là: "cho điểm truy vấn q, những điểm đã lưu nào gần nó nhất?"
#
# Thước đo gần như ai cũng dùng là cosine similarity — góc giữa hai vector,
# bỏ qua độ dài. Vì sao bỏ độ dài? Vì độ lớn của embedding chủ yếu chứa rác
# (độ dài văn bản, đặc thù tần suất token), còn *hướng* mới chứa ngữ nghĩa.


def dot(a: Vector, b: Vector) -> float:
    """Σ aᵢ·bᵢ — vòng lặp trong cùng của mọi vector database trên đời.

    Khi faiss hay pgvector đốt CPU, chúng đốt ở đúng chỗ này, được vector
    hoá bằng AVX-512 với 8-16 số thực mỗi lệnh. Ta làm từng số một — cùng
    một phép tính, chiếu chậm.
    """
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    """Độ dài Euclid ‖a‖ = sqrt(Σ aᵢ²)."""
    return math.sqrt(dot(a, a))


def cosine(a: Vector, b: Vector) -> float:
    """cos(θ) = a·b / (‖a‖·‖b‖) — nằm trong [-1, 1], 1 nghĩa là 'cùng hướng'."""
    return dot(a, b) / (norm(a) * norm(b))


def normalize(a: Vector) -> Vector:
    """Co giãn vector về độ dài 1.

    Mẹo NGHỀ của ngành: nếu normalize mọi vector một lần lúc insert, thì
    cosine(a, b) == dot(a, b) — hai phép nhân mỗi chiều thay vì hai căn bậc
    hai và một phép chia mỗi lần so sánh. Mọi vector DB thật đều làm vậy khi
    bạn chọn khoảng cách cosine. Ta cũng thế: từ đây trở đi, mọi thứ được
    lưu đều có độ dài 1 và 'độ tương đồng' nghĩa là dot().
    """
    n = norm(a)
    if n == 0:
        raise ValueError("cannot index the zero vector (it has no direction)")
    return [x / n for x in a]


# ---------------------------------------------------------------------------
# Chương 2 — Tìm chính xác: brute force, baseline trung thực
# ---------------------------------------------------------------------------
# Chỉ mục đơn giản nhất có thể: giữ một list, so sánh truy vấn với TẤT CẢ,
# trả về top k. Chính xác tuyệt đối. Nhưng cũng O(N·d) mỗi truy vấn:
# với 10 triệu vector × 1536 chiều là ~15 tỷ phép nhân mỗi truy vấn.
# Mọi thứ còn lại trong file này tồn tại vì đoạn văn này.


class BruteForceIndex:
    """Tìm hàng xóm gần nhất chính xác bằng cách quét toàn bộ.

    Không phải bù nhìn đâu! Dưới ~50k vector đây thường là lựa chọn ĐÚNG:
    không tốn công build, không mất recall, đúng đắn một cách hiển nhiên.
    faiss gọi nó là IndexFlat và khuyên dùng nó làm điểm xuất phát.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: list[Vector] = []   # hàng i = vector có id nội bộ i
        self.distance_evals = 0            # sổ sách để test CHỨNG MINH được
                                           # HNSW làm ít việc hơn (Chương 3)

    def __len__(self) -> int:
        return len(self._vectors)

    def add(self, vector: Vector) -> int:
        """Lưu một vector, trả về id nguyên nội bộ (số thứ tự hàng)."""
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        self._vectors.append(normalize(vector))
        return len(self._vectors) - 1

    def search(self, query: Vector, k: int) -> list[tuple[float, int]]:
        """Trả về k cặp (độ tương đồng, id nội bộ) tốt nhất, tốt nhất trước."""
        q = normalize(query)
        self.distance_evals += len(self._vectors)
        scored = ((dot(q, v), i) for i, v in enumerate(self._vectors))
        # heapq.nlargest là top-k O(N log k) — tốt hơn sort cả N phần tử.
        return heapq.nlargest(k, scored)

    # (Lưu trữ cho Chương 5 — chỉ là các hàng thô.)
    def to_dict(self) -> dict:
        return {"dim": self.dim, "vectors": self._vectors}

    @classmethod
    def from_dict(cls, d: dict) -> "BruteForceIndex":
        idx = cls(d["dim"])
        idx._vectors = d["vectors"]        # đã normalize sẵn từ lúc lưu
        return idx


# ---------------------------------------------------------------------------
# Chương 3 — HNSW: chỉ mục bạn dùng hằng ngày nhưng không giải thích được
# ---------------------------------------------------------------------------
# Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016) là chỉ
# mục ANN chủ lực: Qdrant, Weaviate, Milvus, pgvector, Vespa, Redis — toàn
# HNSW. Ý tưởng gói gọn trong ba câu:
#
#   1. Nối mỗi vector với ~M hàng xóm gần nhất → một đồ thị mà bạn có thể
#      "đi bộ" về phía truy vấn bằng cách luôn bước sang hàng xóm gần nó
#      nhất (tìm kiếm tham lam).
#   2. Đồ thị phẳng khiến bước đi tham lam khởi động chậm (nhiều bước nhỏ
#      từ điểm xuất phát ngẫu nhiên) và dễ kẹt ở cực trị địa phương. Nên ta
#      xây các TẦNG, như hệ thống đường cao tốc: tầng 2 có ~1% số nút (bước
#      nhảy dài), tầng 1 có ~10%, tầng 0 có tất cả. Bắt đầu từ đỉnh, nhảy
#      xa, tụt xuống một tầng khi hết cải thiện. Zoom out, rồi zoom in.
#   3. Để chắc chắn hơn, đừng đi với một "điểm tốt nhất hiện tại" duy nhất
#      mà với một chùm `ef` ứng viên. ef lớn = recall cao, tốn công hơn.
#
# Toàn bộ bí kíp chỉ có vậy. Tìm kiếm còn ~O(log N) bước nhảy thay vì O(N)
# phép so sánh, đổi lại nó *xấp xỉ*: có thể bỏ sót hàng xóm thật. Các test
# ở Chương 3 của test_vectordb.py đo chính xác tần suất bỏ sót đó.


class HNSWIndex:
    """Một HNSW gọn mà trung thành với bản gốc. Cùng API với BruteForceIndex."""

    def __init__(self, dim: int, m: int = 16, ef_construction: int = 100,
                 ef_search: int = 64, seed: int = 0):
        self.dim = dim
        self._m = m                    # số liên kết tối đa mỗi nút, tầng ≥ 1
        self._m0 = 2 * m               # tầng 0 dày hơn (mặc định của paper)
        self._ef_construction = ef_construction  # độ rộng chùm khi XÂY
        self._ef_search = ef_search              # độ rộng chùm khi TRUY VẤN
        # Tầng cao nhất của mỗi nút được rút từ phân phối kiểu hình học:
        # P(level ≥ L) = (1/M)^L. Với M=16, ~94% số nút chỉ sống ở tầng 0,
        # ~6% lên tới tầng 1, ~0.4% lên tầng 2... đúng cái dáng "ít sân bay
        # đường dài, nhiều đường làng" mà ta muốn.
        self._level_mult = 1.0 / math.log(m)
        self._rng = random.Random(seed)  # seed cố định → đồ thị y hệt mỗi lần
        self._vectors: list[Vector] = []
        # _neighbors[i][L] = list các id nút nối với nút i ở tầng L.
        self._neighbors: list[list[list[int]]] = []
        self._entry: int | None = None   # id nút nơi mọi truy vấn bắt đầu
        self._max_level = 0              # tầng cao nhất đang có nút
        self.distance_evals = 0

    def __len__(self) -> int:
        return len(self._vectors)

    def _dot(self, q: Vector, node: int) -> float:
        """Mọi phép đo tương đồng đi qua đây để distance_evals luôn trung thực."""
        self.distance_evals += 1
        return dot(q, self._vectors[node])

    def _random_level(self) -> int:
        # -ln(U) là biến ngẫu nhiên mũ; nhân 1/ln(M) rồi lấy phần nguyên
        # cho ra phân phối tầng hình học mô tả ở trên.
        return int(-math.log(self._rng.random()) * self._level_mult)

    # -- thuật toán duy nhất thật sự quan trọng: beam search tham lam 1 tầng --
    def _search_layer(self, q: Vector, entry: list[int], ef: int,
                      layer: int) -> list[tuple[float, int]]:
        """Từ các nút `entry`, đi trên tầng `layer` về phía q với chùm rộng ef.

        Hai heap, một bất biến:
          * `candidates`: biên giới còn phải mở rộng, tương đồng cao nhất trước.
          * `results`:    ef nút tốt nhất từng thấy (min-heap, nên phần tử
                          TỆ nhất trong nhóm tốt nằm ở results[0], loại rẻ).
        Dừng khi ứng viên chưa mở tốt nhất còn tệ hơn kết quả tệ nhất —
        biên giới hết khả năng cải thiện đáp án. Cú thoát sớm này là lý do
        HNSW nhanh; phần còn lại chỉ là sổ sách.
        """
        sims = {e: self._dot(q, e) for e in entry}
        visited = set(entry)
        candidates = [(-s, e) for e, s in sims.items()]  # max-heap nhờ dấu trừ
        heapq.heapify(candidates)
        results = [(s, e) for e, s in sims.items()]      # min-heap, tự nhiên
        heapq.heapify(results)
        while candidates:
            neg_sim, node = heapq.heappop(candidates)
            if len(results) >= ef and -neg_sim < results[0][0]:
                break                        # biên giới hết cửa thắng
            for nb in self._neighbors[node][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                s = self._dot(q, nb)
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, nb))
                    heapq.heappush(results, (s, nb))
                    if len(results) > ef:
                        heapq.heappop(results)   # loại phần tử tệ nhất hiện tại
        return sorted(results, reverse=True)     # tốt nhất trước

    def add(self, vector: Vector) -> int:
        """Insert = tìm chỗ vector thuộc về, rồi nối nó vào đồ thị."""
        if len(vector) != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {len(vector)}")
        q = normalize(vector)
        new = len(self._vectors)
        self._vectors.append(q)
        level = self._random_level()
        self._neighbors.append([[] for _ in range(level + 1)])

        if self._entry is None:              # nút đầu tiên: chưa có gì để nối
            self._entry, self._max_level = new, level
            return new

        # Giai đoạn 1: từ đỉnh đồ thị xuống ngay trên tầng của nút mới,
        # tụt dần kiểu tham lam với chùm rộng 1 (ta chỉ cần một điểm vào
        # tốt, chưa cần cả tập kết quả).
        ep = self._entry
        for layer in range(self._max_level, level, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]

        # Giai đoạn 2: trên mỗi tầng nút mới có mặt, tìm đúng khu hàng xóm
        # bằng chùm xây dựng rộng, rồi đấu dây.
        for layer in range(min(level, self._max_level), -1, -1):
            found = self._search_layer(q, [ep], self._ef_construction, layer)
            cap = self._m0 if layer == 0 else self._m
            selected = [node for _, node in found[: self._m]]
            self._neighbors[new][layer] = selected
            for nb in selected:
                links = self._neighbors[nb][layer]
                links.append(new)             # liên kết hai chiều
                if len(links) > cap:
                    # Hàng xóm quá tải → chỉ giữ `cap` liên kết gần nó
                    # nhất. (Paper có heuristic đa dạng hoá thông minh hơn
                    # ở đây — xem danh sách "chương bổ sung cần viết".)
                    v = self._vectors[nb]
                    links.sort(key=lambda o: self._dot(v, o), reverse=True)
                    del links[cap:]
            ep = found[0][1]                  # kết quả tốt nhất mồi cho tầng dưới

        if level > self._max_level:           # nút mới cao nhất →
            self._entry = new                 # nó thành cửa vào toàn cục
            self._max_level = level
        return new

    def search(self, query: Vector, k: int,
               ef: int | None = None) -> list[tuple[float, int]]:
        """Top-k (độ tương đồng, id nội bộ): tụt các tầng cao tốc với chùm 1,
        rồi càn quét tầng 0 với chùm đầy đủ. ef là núm vặn recall-vs-tốc-độ
        ngay lúc chạy — đúng cái núm bạn chỉnh ở DB production."""
        if self._entry is None:
            return []
        ef = max(ef or self._ef_search, k)
        q = normalize(query)
        ep = self._entry
        for layer in range(self._max_level, 0, -1):
            ep = self._search_layer(q, [ep], 1, layer)[0][1]
        return self._search_layer(q, [ep], ef, 0)[:k]

    # -- lưu trữ (Chương 5): cả chỉ mục chỉ là 5 trường dữ liệu trơn --------
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
# Chương 4 — Lớp vỏ database: thứ biến chỉ mục thành database
# ---------------------------------------------------------------------------
# Chỉ mục ánh xạ vector sang số nguyên. Một DATABASE nói ngôn ngữ của bạn:
# ID chuỗi, metadata, filter, upsert, delete. Lớp bọc này không hào nhoáng —
# và ở các vector DB thật, nó cũng chiếm phần lớn code y như vậy.
#
# Cái gai thú vị: HNSW KHÔNG XOÁ ĐƯỢC. Gỡ một nút sẽ xé thủng đồ thị và
# tìm kiếm tham lam rơi vào hố. Nên, giống Qdrant và bạn bè, ta dùng
# TOMBSTONE: đánh dấu hàng đã chết, bỏ qua nó trong kết quả, và lấy dư để
# bù. (DB thật cuối cùng sẽ xây lại đồ thị để dọn tombstone — đó chính là
# nút "vacuum"/"optimize".)


class VectorDB:
    """Database hướng người dùng: upsert / query / delete / save / load."""

    _INDEXES = {"hnsw": HNSWIndex, "flat": BruteForceIndex}

    def __init__(self, dim: int, index: str = "hnsw", **index_params):
        self.dim = dim
        self._index_type = index
        self._index = self._INDEXES[index](dim, **index_params)
        self._row_of: dict[str, int] = {}   # id người dùng -> hàng nội bộ
        self._id_of: dict[int, str] = {}    # hàng nội bộ -> id người dùng
        self._meta: dict[str, dict] = {}    # id người dùng -> metadata
        self._dead: set[int] = set()        # các hàng nội bộ đã tombstone

    def __len__(self) -> int:
        return len(self._row_of)

    def upsert(self, id: str, vector: Vector, metadata: dict | None = None):
        """Thêm hoặc thay thế. 'Thay thế' là lời nói dối mà mọi DB chạy HNSW
        đều nói: hàng cũ bị tombstone và vector được chèn như nút MỚI, vì
        đồ thị không đấu lại dây tại chỗ được. Hãy nhìn len(self._dead)
        phình lên khi bạn upsert cùng một id trong vòng lặp — đó là 'index
        bloat' ngoài đời thật."""
        if id in self._row_of:
            self._dead.add(self._row_of[id])
        row = self._index.add(vector)
        self._row_of[id] = row
        self._id_of[row] = id
        self._meta[id] = metadata or {}

    def delete(self, id: str):
        """Chỉ tombstone — O(1), không phẫu thuật đồ thị. Xem mở đầu chương."""
        self._dead.add(self._row_of.pop(id))
        self._meta.pop(id)

    def query(self, vector: Vector, k: int = 5,
              where: dict | None = None) -> list[dict]:
        """Top-k hàng còn sống dạng {id, score, metadata}, lọc theo `where`
        (khớp chính xác trên các khoá metadata — kiểu filter 80% trường hợp).

        Đây là POST-filtering: tìm trước, vứt các kết quả trượt filter sau.
        Để sống sót qua đợt vứt đó ta lấy dư — k cộng thêm theo số tombstone
        và một hệ số phòng hờ. Đây là mánh trung thực mà đa số DB khởi đầu;
        điểm chết của nó (filter chọn lọc cao đòi lấy dư khổng lồ) chính là
        lý do 'filtered ANN' vẫn là đề tài nghiên cứu sôi nổi. Chương về
        lọc trong đồ thị kiểu Qdrant sẽ lắp vừa khít ngay chỗ này.
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
    # Chương 5 — Lưu trữ: database phải sống sót qua một lần khởi động lại
    # -----------------------------------------------------------------------
    # Ta serialize TẤT CẢ — vector, liên kết đồ thị, bảng id, tombstone —
    # vào một file JSON duy nhất. Chọn JSON để bạn có thể `cat` database
    # của mình ra và thấy bên trong không có phép màu nào. Engine thật khác
    # về mức độ, không khác về bản chất: layout nhị phân, mmap để HĐH nạp
    # file theo trang một cách lười biếng, và write-ahead log để crash giữa
    # chừng không làm hỏng file. Vẫn là năm trường dữ liệu đó.

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
        db = cls.__new__(cls)                # bỏ qua __init__; khôi phục field
        db.dim = state["dim"]
        db._index_type = state["index_type"]
        db._index = cls._INDEXES[db._index_type].from_dict(state["index"])
        db._row_of = state["row_of"]
        db._id_of = {row: id for id, row in db._row_of.items()}
        db._meta = state["meta"]
        db._dead = set(state["dead"])
        return db


# ---------------------------------------------------------------------------
# Phần kết — demo không cần dependency nào (kể cả embeddings, thật đấy)
# ---------------------------------------------------------------------------
# Embedding thật đến từ mạng nơ-ron. Nhưng BẤT KỲ hàm text→vector nào đưa
# các văn bản giống nhau về gần nhau đều demo được bộ máy này. Trigram ký tự
# băm nhồi vào bucket là model embedding tệ nhất thế giới — và hoàn toàn
# trong suốt, mà tối nay thì đó mới là cái đáng đánh đổi.


def embed(text: str, dim: int = 64) -> Vector:
    """Túi trigram ký tự, feature-hash vào `dim` bucket."""
    v = [0.0] * dim
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = zlib.crc32(t[i:i + 3].encode())     # crc32: ổn định giữa các lần
        v[h % dim] += 1.0 if h & 1 else -1.0    # chạy, khác hash() của Python
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
