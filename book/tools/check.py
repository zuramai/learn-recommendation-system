#!/usr/bin/env python3
"""Validate the book sources (plan 9.8).

    uv run python tools/check.py --lang en|id|all [--src DIR]

Exits non-zero and lists every violation. Fact boxes: must contain a year in
parentheses `(2003)` or a `Source: Author, Venue, Year` line; the first author
(first capitalised name of the first comma/&/and-separated part of the Source line)
must appear in `93-bibliography.md`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build  # noqa: E402

Q_START = re.compile(r"^\*\*Q(\d+)\.(\d+)\*\*")
SUMMARY_RE = re.compile(r"^(summary|ringkasan)$", re.I)
QUIZ_RE = re.compile(r"^(quiz|kuis)$", re.I)
SOURCE_RE = re.compile(r"^\s*(?:\*\*)?(?:Source|Sumber)(?:\*\*)?\s*:\s*(.+)$", re.I | re.M)
YEAR_RE = re.compile(r"\((?:[^()]*?\b)?(1[89]\d\d|20\d\d)\b[^()]*\)")


def fact_boxes(tokens) -> list[tuple[int, str]]:
    """(token index, plain text) of every ::: fact container."""
    out = []
    for i, t in enumerate(tokens):
        if t.type == "container_fact_open":
            parts, j = [], i + 1
            while j < len(tokens) and tokens[j].type != "container_fact_close":
                if tokens[j].type == "inline":
                    parts.append(tokens[j].content)
                elif tokens[j].type == "fence":
                    parts.append(tokens[j].content)
                j += 1
            out.append((i, "\n".join(parts)))
    return out


def check_chapter(ch, lang: str, biblio: str | None) -> tuple[list[str], set[str]]:
    errs: list[str] = []
    ids: set[str] = set()
    name = ch.path.name
    err = lambda m: errs.append(f"{name}: {m}")  # noqa: E731
    toks = ch.tokens
    for e in ch.errors:
        err(e)

    # includes / outputs / images (all chapters)
    for t in toks:
        if t.type == "fence":
            info = t.info.strip().split()[0] if t.info.strip() else ""
            first = t.content.strip().splitlines()[0] if t.content.strip() else ""
            try:
                if info == "include":
                    build.resolve_include(t.content)
                elif info == "output":
                    build.resolve_output(t.content)
            except (ValueError, SyntaxError) as e:
                err(f"{info} ({first}): {e}")
        if t.type == "inline":
            for c in t.children or []:
                if c.type == "image":
                    src = c.attrGet("src") or ""
                    if not re.match(r"[a-z]+://", src) and not (ch.path.parent / src).is_file():
                        err(f"image not found: {src}")

    h2 = [(i, toks[i + 1].content.strip()) for i, t in enumerate(toks)
          if t.type == "heading_open" and t.tag == "h2"]
    facts = fact_boxes(toks)

    # fact box sourcing (any file)
    for _, text in facts:
        m = SOURCE_RE.search(text)
        if not YEAR_RE.search(text) and not m:
            err(f"fact box without a year in parentheses or a Source: line: {text[:60]!r}")
        if m:
            if biblio is None:
                err("fact box cites a source but 93-bibliography.md is missing")
                continue
            first_part = re.split(r",|&| and | dan |\bet al\b|\(", m[1])[0]
            names = re.findall(r"[A-Z][\w'’\-]+", first_part)
            if names and not any(n.lower() in biblio for n in names):
                err(f"fact source {first_part.strip()!r} not found in 93-bibliography.md")

    if ch.kind != "chapter":
        return errs, ids

    n = ch.index
    if not (1 <= n <= 14):
        return errs, ids

    # facts placement
    if len(facts) < 2:
        err(f"needs >= 2 ::: fact blocks (found {len(facts)})")
    first_h2 = h2[0][0] if h2 else None
    summary_idx = next((i for i, text in h2 if SUMMARY_RE.match(text)), None)
    if summary_idx is None:
        err("missing `## Summary` / `## Ringkasan` heading")
    for i, _ in facts:
        if first_h2 is None or i < first_h2:
            err("fact box before the first `##` heading")
        if summary_idx is not None and i > summary_idx:
            err("fact box after the Summary heading")

    # mermaid / images
    if not any(t.type == "fence" and t.info.strip() == "mermaid" for t in toks):
        err("needs >= 1 mermaid block")
    n_img = sum(1 for t in toks if t.type == "inline" for c in (t.children or []) if c.type == "image")
    if n_img < 2:
        err(f"needs >= 2 images (found {n_img})")

    # quiz
    if not h2 or not QUIZ_RE.match(h2[-1][1]):
        err("last `##` heading must be Quiz / Kuis")
    else:
        qid: list[tuple[int, int]] = []
        for t in toks[h2[-1][0]:]:
            if t.type == "inline":
                m = Q_START.match(t.content.strip())
                if m:
                    qid.append((int(m[1]), int(m[2])))
        if not 8 <= len(qid) <= 10:
            err(f"quiz must have 8-10 questions (found {len(qid)})")
        if any(c != n for c, _ in qid):
            err(f"quiz ids must be Q{n}.<n>")
        if [q for _, q in qid] != list(range(1, len(qid) + 1)):
            err(f"quiz ids must be numbered consecutively from 1 (found {[f'{c}.{q}' for c, q in qid]})")
        ids = {f"Q{c}.{q}" for c, q in qid}

    # prose words (EN only)
    if lang == "en":
        words = sum(len(re.findall(r"\w+", t.content)) for t in toks if t.type == "inline")
        if words < 3000:
            err(f"only {words} prose words (need >= 3000)")
    return errs, ids


def check_dir(src: Path, lang: str) -> list[str]:
    chapters = build.load_chapters(src, lang)
    errs: list[str] = []
    if not chapters:
        return [f"no chapter files in {src}"]
    bib_path = src / "93-bibliography.md"
    biblio = bib_path.read_text().lower() if bib_path.is_file() else None
    chapter_ids: set[str] = set()
    for ch in chapters:
        e, ids = check_chapter(ch, lang, biblio)
        errs += e
        chapter_ids |= ids
    ans_path = next(iter(sorted(src.glob("90-*.md"))), None)
    if ans_path is None:
        if chapter_ids:
            errs.append("appendix A (90-*.md) missing: quiz answers not found")
    else:
        ans_ch = next(c for c in chapters if c.path == ans_path)
        ans = set()
        for t in ans_ch.tokens:
            if t.type == "inline":
                m = Q_START.match(t.content.strip())
                if m:
                    ans.add(f"Q{m[1]}.{m[2]}")
        for q in sorted(chapter_ids - ans):
            errs.append(f"{ans_path.name}: no answer for {q}")
        for q in sorted(ans - chapter_ids):
            errs.append(f"{ans_path.name}: answer for unknown question {q}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "id", "all"], default="all")
    ap.add_argument("--src", type=Path, help="chapter directory override (single --lang only)")
    args = ap.parse_args()
    langs = ["en", "id"] if args.lang == "all" else [args.lang]
    if args.src and len(langs) != 1:
        ap.error("--src needs a single --lang")
    bad = False
    for lang in langs:
        errs = check_dir(args.src or build.BOOK / lang, lang)
        if errs:
            bad = True
            print(f"[{lang}] {len(errs)} violation(s):")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"[{lang}] OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
