"""Verify a translation changed only comments/docstrings, never code.

Usage: python3 tools/check_translation.py translations/vi/vectordb.py

The contract for translations: identical code, translated prose. We enforce
it mechanically — tokenize both files, drop comments and docstrings, and
require the remaining token streams to match exactly.
"""

import io
import sys
import token
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def code_tokens(path: Path) -> list[tuple[int, str]]:
    toks = []
    prev_meaningful = tokenize.INDENT  # docstrings are STRINGs at stmt start
    with open(path, "rb") as f:
        for t in tokenize.tokenize(f.readline):
            if t.type in (token.COMMENT, token.NL, token.NEWLINE,
                          token.INDENT, token.DEDENT, token.ENCODING):
                if t.type == token.NEWLINE:
                    prev_meaningful = t.type
                continue
            # A STRING right after a NEWLINE/start is a docstring: skip it.
            if t.type == token.STRING and prev_meaningful in (
                    tokenize.NEWLINE, tokenize.INDENT):
                prev_meaningful = t.type
                continue
            prev_meaningful = t.type
            toks.append((t.type, t.string))
    return toks


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    translated = Path(sys.argv[1])
    original = ROOT / translated.name
    a, b = code_tokens(original), code_tokens(translated)
    if a == b:
        print(f"ok: {translated} — code identical to {original.name}, "
              f"only prose differs")
        return 0
    for i, (ta, tb) in enumerate(zip(a, b)):
        if ta != tb:
            print(f"MISMATCH at code token #{i}: {ta} != {tb}")
            break
    else:
        print(f"MISMATCH: token counts differ ({len(a)} vs {len(b)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
