# Contributing

Two contribution tracks. Both share one goal: **someone reads it once and
finally understands.**

## Track 1 — Translations

Make the book readable in your language.

1. Copy `vectordb.py` to `translations/<lang>/vectordb.py` (ISO 639-1 code:
   `vi`, `ja`, `es`, ...).
2. Translate the module docstring, all comments, and all docstrings.
   **Do not touch the code** — not even identifier names. The contract is:
   same program, different prose.
3. Verify mechanically:
   ```bash
   python3 tools/check_translation.py translations/<lang>/vectordb.py
   ```
4. Optionally translate `README.md` → `translations/<lang>/README.md` and
   `test_vectordb.py` the same way. A README-only PR is a fine start.

## Track 2 — Extra chapters

Add a chapter the main file deliberately left out (wishlist in the README).

Conventions:

- One file: `extras/NN_topic.py` (next free number), with tests either
  inline under `if __name__ == "__main__"` or in `extras/test_NN_topic.py`.
- **Stdlib only.** If it needs numpy, it needs rewriting.
- Self-contained: importing from `vectordb.py` is fine; extras must not
  import each other.
- Comment density like the main file: a reader who knows Python but not the
  topic should never have to leave the file.
- Tests teach: each test docstring states the claim the assertion proves.
  Keep the whole suite under ~5 seconds.
- Target length: 100–250 lines. If it wants to be longer, it's two chapters.

## Ground rules

- `python3 test_vectordb.py` must stay green, dependency-free, and fast.
- The main file stays ~500 lines. Improvements to its clarity are welcome;
  new features belong in `extras/`.
