"""
test_vectordb.py — tour có hướng dẫn.

Mỗi test là một luận điểm chương trong vectordb.py, phát biểu lại thành
assertion. Hãy đọc theo thứ tự; chúng xây tiếp lên nhau. Chạy bằng Python
thuần (`python test_vectordb.py`) hoặc pytest — cả hai đều không cần cài gì.
"""

import math
import os
import random
import tempfile

from vectordb import (BruteForceIndex, HNSWIndex, VectorDB, cosine, dot,
                      embed, normalize)

# ---------------------------------------------------------------------------
# Bộ dữ liệu đồ chơi dùng chung: 2000 vector 16 chiều ngẫu nhiên. Các thành
# phần Gaussian cho hướng phân bố đều trên mặt cầu — trường hợp khó nhất,
# trung thực nhất với ANN (embedding thật dễ hơn: chúng sống trên các
# manifold ít chiều).
# ---------------------------------------------------------------------------
N, DIM = 2000, 16
_rng = random.Random(42)
DATASET = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(N)]
QUERIES = [[_rng.gauss(0, 1) for _ in range(DIM)] for _ in range(20)]

_built = {}  # mỗi chỉ mục chỉ build một lần, các test dùng chung


def build(kind):
    if kind not in _built:
        idx = {"flat": BruteForceIndex, "hnsw": HNSWIndex}[kind](DIM)
        for v in DATASET:
            idx.add(v)
        _built[kind] = idx
    return _built[kind]


# ---------------------------------------------------------------------------
# Chương 1 — Độ tương đồng
# ---------------------------------------------------------------------------

def test_ch1_cosine_measures_direction_not_length():
    """cosine(a, b) chỉ quan tâm GÓC giữa hai vector: một vector tương đồng
    tối đa với chính nó (1.0) và với mọi bản sao co giãn của nó — chính là
    lý do vứt bỏ độ lớn là an toàn."""
    a = [1.0, 2.0, 3.0]
    assert math.isclose(cosine(a, a), 1.0)
    assert math.isclose(cosine(a, [10.0, 20.0, 30.0]), 1.0)   # cùng hướng
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0)          # vuông góc
    assert math.isclose(cosine([1, 0], [-1, 0]), -1.0)        # ngược hướng


def test_ch1_the_normalization_trick():
    """Sau khi normalize cả hai vector, dot() trơn bằng đúng cosine() —
    đẳng thức này là lý do mọi DB normalize lúc insert và không bao giờ
    tính một phép cosine thật nào nữa."""
    a, b = [3.0, -1.0, 2.0], [0.5, 4.0, -2.0]
    assert math.isclose(dot(normalize(a), normalize(b)), cosine(a, b))


def test_ch1_zero_vector_is_rejected():
    """Vector không có hướng nào cả, nên độ tương đồng cosine với nó là
    vô nghĩa (0/0). Database tốt từ chối nó ầm ĩ ngay ở cửa thay vì lặng
    lẽ trả NaN lúc truy vấn."""
    try:
        normalize([0.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Chương 2 — Tìm chính xác
# ---------------------------------------------------------------------------

def test_ch2_brute_force_is_exact():
    """Brute force trả về các hàng xóm gần nhất THẬT — ta kiểm chứng bằng
    một lần sort toàn bộ độc lập. Đây là chân lý gốc để chấm điểm HNSW."""
    idx = build("flat")
    q = QUERIES[0]
    got = [i for _, i in idx.search(q, 10)]
    qn = normalize(q)
    want = sorted(range(N), key=lambda i: dot(qn, normalize(DATASET[i])),
                  reverse=True)[:10]
    assert got == want


def test_ch2_brute_force_cost_is_the_whole_dataset():
    """Đề bài, phát biểu thành assertion: mỗi truy vấn chạm vào mọi vector.
    Hôm nay là 2000; ở quy mô production là 800 triệu. Test này là động cơ
    của toàn bộ chương kế tiếp."""
    idx = build("flat")
    before = idx.distance_evals
    idx.search(QUERIES[1], 10)
    assert idx.distance_evals - before == N


# ---------------------------------------------------------------------------
# Chương 3 — HNSW
# ---------------------------------------------------------------------------

def recall_at_10(hnsw, flat, queries, ef=None):
    """Recall@10 = |top-10 của HNSW ∩ top-10 thật| / 10, trung bình trên các
    truy vấn. Thước đo VÀNG của giới ANN — mọi benchmark bạn từng thấy đều
    vẽ nó."""
    total = 0.0
    for q in queries:
        truth = {i for _, i in flat.search(q, 10)}
        found = {i for _, i in hnsw.search(q, 10, ef=ef)}
        total += len(truth & found) / 10
    return total / len(queries)


def test_ch3_hnsw_finds_almost_everything():
    """'Xấp xỉ' là thứ đo được, không phải nói suông: trên vector ngẫu nhiên
    phân bố đều (trường hợp xấu nhất), HNSW với cấu hình mặc định vẫn phải
    tìm thấy ≥90% của top-10 thật."""
    r = recall_at_10(build("hnsw"), build("flat"), QUERIES)
    assert r >= 0.90, f"recall@10 = {r:.2f}"


def test_ch3_hnsw_does_a_fraction_of_the_work():
    """Phần thưởng: HNSW trả lời truy vấn với số phép đo khoảng cách ít hơn
    hẳn con số N=2000 mà brute force cần. Khoảng cách này CÀNG RỘNG khi N
    tăng — brute force leo thang O(N), còn cuộc dạo đồ thị ~O(log N)."""
    idx = build("hnsw")
    before = idx.distance_evals
    for q in QUERIES:
        idx.search(q, 10)
    per_query = (idx.distance_evals - before) / len(QUERIES)
    assert per_query < N * 0.6, f"{per_query:.0f} evals vs {N} brute force"


def test_ch3_ef_is_the_recall_speed_dial():
    """Vặn ef xuống → nhanh hơn nhưng mù hơn. Vặn lên → chậm hơn nhưng gần
    như chính xác. Một cái núm này là phần lớn ý nghĩa của cụm từ 'tune
    vector DB của bạn'."""
    hnsw, flat = build("hnsw"), build("flat")
    assert recall_at_10(hnsw, flat, QUERIES, ef=200) >= \
           recall_at_10(hnsw, flat, QUERIES, ef=10) - 0.01  # biên độ nhiễu


def test_ch3_the_graph_is_a_hierarchy():
    """Trò xổ số tầng đã chạy đúng: tầng 0 chứa mọi nút, và mỗi tầng cao hơn
    chứa một mẫu mỏng dần theo cấp số nhân (cái dáng 'đường cao tốc')."""
    idx = build("hnsw")
    sizes = [sum(1 for nb in idx._neighbors if len(nb) > L)
             for L in range(idx._max_level + 1)]
    assert sizes[0] == N
    assert all(a > b for a, b in zip(sizes, sizes[1:]))


# ---------------------------------------------------------------------------
# Chương 4 — Lớp vỏ database
# ---------------------------------------------------------------------------

def make_db():
    db = VectorDB(dim=64)
    db.upsert("fern", embed("water your fern weekly"), {"topic": "plants"})
    db.upsert("cactus", embed("a cactus barely needs water"), {"topic": "plants"})
    db.upsert("git", embed("git reset undoes a commit"), {"topic": "code"})
    return db


def test_ch4_query_speaks_ids_and_metadata():
    """Lớp vỏ làm phiên dịch: bạn không bao giờ thấy số hàng nội bộ, chỉ
    thấy id và payload của chính mình, xếp hạng theo điểm."""
    hits = make_db().query(embed("how often to water a fern"), k=2)
    assert hits[0]["id"] == "fern"
    assert hits[0]["metadata"] == {"topic": "plants"}
    assert hits[0]["score"] >= hits[1]["score"]


def test_ch4_where_filters_but_search_still_ranks():
    """Filter thay đổi AI đủ điều kiện, không thay đổi CÁCH xếp hạng: hỏi
    một câu về cây cối với where={'topic': 'code'} thì chỉ được trả về tài
    liệu code."""
    hits = make_db().query(embed("how often to water a fern"), k=5,
                           where={"topic": "code"})
    assert [h["id"] for h in hits] == ["git"]


def test_ch4_delete_is_a_tombstone():
    """Hàng bị xoá biến mất khỏi kết quả ngay lập tức — nhưng vector vẫn nằm
    trong đồ thị (len(_dead) đã tăng). Khoảng hở giữa 'mất với bạn' và 'mất
    thật sự' chính là pattern tombstone."""
    db = make_db()
    db.delete("fern")
    assert "fern" not in [h["id"] for h in
                          db.query(embed("water a fern"), k=5)]
    assert len(db._dead) == 1


def test_ch4_upsert_replaces_and_bloats():
    """Upsert lại một id sẽ phục vụ vector/metadata MỚI, đồng thời lặng lẽ
    tombstone hàng cũ — index bloat quan sát được, lý do các DB thật có nút
    'optimize'."""
    db = make_db()
    db.upsert("fern", embed("ferns love humid bathrooms"), {"topic": "plants",
                                                            "v": 2})
    hits = db.query(embed("humid bathroom plant"), k=1)
    assert hits[0]["id"] == "fern" and hits[0]["metadata"]["v"] == 2
    assert len(db) == 3 and len(db._dead) == 1


# ---------------------------------------------------------------------------
# Chương 5 — Lưu trữ
# ---------------------------------------------------------------------------

def test_ch5_a_database_survives_a_restart():
    """Save, 'crash', load: database tái sinh cho câu trả lời giống hệt từng
    byte — cùng id, cùng điểm, cùng sự tôn trọng dành cho tombstone."""
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
# Bộ chạy Python thuần (pytest cũng tự nhặt được các test này)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} tests passed.")
