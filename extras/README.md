# Extra chapters

Community-contributed chapters live here — one self-contained, stdlib-only,
heavily annotated file each. Wishlist and conventions: see
[CONTRIBUTING.md](../CONTRIBUTING.md) and the README.

| Chapter | File | Tests |
|---|---|---|
| 6 — IVF: k-means cells + `nprobe` (and a real `delete()`) | [`06_ivf.py`](06_ivf.py) | [`test_06_ivf.py`](test_06_ivf.py) |
| 7 — Product quantization: bytes for vectors, ADC, rerank | [`07_pq.py`](07_pq.py) | [`test_07_pq.py`](test_07_pq.py) |

Run any chapter directly (`python3 extras/06_ivf.py`) for its demo; numeric
filename prefixes mean tests load chapters by path with `importlib` — copy
the header of an existing test file to start yours.
