# Xây vector database từ đầu, trong ~500 dòng Python

Bạn dùng vector database mỗi ngày — mọi pipeline RAG, mọi app "chat với tài
liệu", mọi ô tìm kiếm ngữ nghĩa. Nhưng hỏi một kỹ sư *nó thực sự làm gì*,
câu trả lời thường là: "ờ… nó tìm các vector giống nhau. Nhanh."

Repo này là lời giải thích còn thiếu. Một file duy nhất,
[`vectordb.py`](../../vectordb.py) (466 dòng, khoảng một nửa là chú thích),
chứa một vector database thật sự chạy được — thuần Python chuẩn, không
numpy, không dependency nào cả.

## Các chương

| Chương | Bạn sẽ hiểu được gì |
|---|---|
| 1 — Độ tương đồng | Vì sao dùng cosine, và mẹo normalize-một-lần mọi DB thật đều dùng |
| 2 — Tìm kiếm chính xác | Brute force: đúng 100%, O(N·d) — chính là bài toán cần giải |
| 3 — HNSW | Chỉ mục đồ thị bên trong Qdrant/Weaviate/pgvector/Milvus, ~150 dòng |
| 4 — Lớp vỏ database | ID, metadata, filter — và vì sao HNSW *không xoá được* (tombstone!) |
| 5 — Lưu trữ | Cả database là một file JSON bạn có thể `cat` ra xem |

## Chạy thử

```bash
python3 vectordb.py        # demo 30 giây, không cần cài gì
python3 test_vectordb.py   # tour có hướng dẫn (pytest cũng chạy được)
```

## Trạng thái bản dịch

- [x] README (file này)
- [x] [`vectordb.py`](vectordb.py) — chú thích đã dịch, code giữ nguyên
      từng token (kiểm chứng bởi CI qua
      `python3 tools/check_translation.py translations/vi/vectordb.py`)
- [x] [`test_vectordb.py`](test_vectordb.py) — bản dịch hoàn chỉnh; CI chạy
      cả bộ test tiếng Việt

Bản dịch tiếng Việt đã hoàn tất — dùng nó làm mẫu cho ngôn ngữ của bạn!

Đóng góp bản dịch rất được hoan nghênh — xem
[CONTRIBUTING.md](../../CONTRIBUTING.md).
