#!/usr/bin/env python3
"""Build the book: Markdown -> HTML -> PDF (headless Chromium via Playwright).

    uv run python tools/build.py --lang en|id|all [--src DIR] [--out PDF] [--html-only]

Conventions (see plan 9.2): ```include / ```output fences, ::: fact|tip|warning|try
callouts, $math$, ```mermaid blocks, numbered figures, `**Q<ch>.<n>**` quiz ids.
`check.py` reuses `load_chapters` / `resolve_include` / `resolve_output` from here.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import html
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

TOOLS = Path(__file__).resolve().parent
BOOK = TOOLS.parent
ROOT = BOOK.parent

I18N = {
    "en": {
        "title": "Recommender Systems: From Scratch to Shopee Scale",
        "subtitle": "A hands-on guide with a working marketplace",
        "built": "Built",
        "toc": "Contents",
        "chapter": "Chapter",
        "appendix": "Appendix",
        "figure": "Figure",
        "fact": "Did you know?",
        "tip": "Tip",
        "warning": "Warning",
        "try": "Try it yourself",
        "output": "Terminal output",
        "lang": "en",
    },
    "id": {
        "title": "Sistem Rekomendasi: Dari Nol hingga Skala Shopee",
        "subtitle": "Panduan praktik dengan marketplace yang berjalan",
        "built": "Dibuat",
        "toc": "Daftar isi",
        "chapter": "Bab",
        "appendix": "Lampiran",
        "figure": "Gambar",
        "fact": "Tahukah kamu?",
        "tip": "Tips",
        "warning": "Peringatan",
        "try": "Coba sendiri",
        "output": "Keluaran terminal",
        "lang": "id",
    },
}
CALLOUTS = ("fact", "tip", "warning", "try")
LANG_BY_EXT = {".py": "python", ".rs": "rust", ".ts": "typescript", ".vue": "vue",
               ".sql": "sql", ".proto": "protobuf", ".toml": "toml"}
CAPTION_RE = re.compile(r"^\*(?:Figure|Gambar|Fig\.?): (.+)\*$", re.S)


class BuildError(Exception):
    pass


# --------------------------------------------------------------------------- includes
def parse_directive(content: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"bad directive line {line!r} (expected `key: value`)")
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _py_symbol(source: str, symbol: str) -> str:
    tree = ast.parse(source)
    scope: list[ast.stmt] = tree.body
    node = None
    for part in symbol.split("."):
        node = next((n for n in scope if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            raise ValueError(f"symbol {symbol!r} not found")
        scope = node.body
    assert node is not None
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = source.splitlines()[first - 1: node.end_lineno]
    return textwrap.dedent("\n".join(lines))


def _region(source: str, name: str) -> str:
    lines = source.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if f"book:start {name}" in line and start is None:
            start = i
        elif f"book:end {name}" in line and start is not None:
            end = i
            break
    if start is None:
        raise ValueError(f"region {name!r}: no `book:start {name}` marker")
    if end is None:
        raise ValueError(f"region {name!r}: no `book:end {name}` marker")
    return textwrap.dedent("\n".join(lines[start + 1: end]))


def resolve_include(content: str) -> tuple[str, str, str]:
    """-> (code, language, label). Raises ValueError with a message on failure."""
    d = parse_directive(content)
    if "path" not in d:
        raise ValueError("missing `path:`")
    path = ROOT / d["path"]
    if not path.is_file():
        raise ValueError(f"file not found: {d['path']}")
    source = path.read_text()
    label = d["path"]
    if "symbol" in d:
        if path.suffix != ".py":
            raise ValueError("`symbol:` only works for .py files (use `region:`)")
        code = _py_symbol(source, d["symbol"])
        label += f" · {d['symbol']}"
    elif "region" in d:
        code = _region(source, d["region"])
        label += f" · {d['region']}"
    else:
        code = source.rstrip("\n")
    lang = d.get("lang") or LANG_BY_EXT.get(path.suffix, "text")
    return code, lang, label


def resolve_output(content: str, base: Path = BOOK) -> str:
    d = parse_directive(content)
    if "file" not in d:
        raise ValueError("missing `file:`")
    path = base / d["file"]
    if not path.is_file():
        raise ValueError(f"output file not found: {d['file']}")
    text = path.read_text().rstrip("\n")
    if "lines" in d:
        m = re.fullmatch(r"(\d+)-(\d+)", d["lines"])
        if not m:
            raise ValueError(f"bad `lines:` {d['lines']!r} (expected A-B)")
        a, b = int(m[1]), int(m[2])
        all_lines = text.splitlines()
        if a < 1 or b < a or a > len(all_lines):
            raise ValueError(f"`lines: {d['lines']}` outside file ({len(all_lines)} lines)")
        text = "\n".join(all_lines[a - 1: b])
    return text


# --------------------------------------------------------------------------- markdown
def highlight_code(code: str, lang: str) -> str:
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = TextLexer()
    return highlight(code, lexer, HtmlFormatter(cssclass="highlight", nowrap=False))


def make_md(L: dict[str, str]) -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    dollarmath_plugin(
        md, allow_space=False, allow_digits=False,
        renderer=lambda content, opts: (
            f"\\[{html.escape(content)}\\]" if opts.get("display_mode")
            else f"\\({html.escape(content)}\\)"),
    )

    def callout(self, tokens, idx, options, env):  # noqa: ANN001
        tok = tokens[idx]
        if tok.nesting == 1:
            name = tok.info.strip().split()[0]
            return (f'<div class="callout {name}"><div class="callout-title">'
                    f'{html.escape(L[name])}</div>\n')
        return "</div>\n"

    for name in CALLOUTS:
        container_plugin(md, name, render=callout)
    return md


@dataclass
class Chapter:
    path: Path
    index: int          # leading number from the file name
    kind: str           # preface | chapter | appendix
    label: str          # "Chapter 3" / "Appendix A" / ""
    fig_prefix: str     # figure number prefix: "3", "P", "A"
    title: str = ""
    tokens: list[Token] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def classify(path: Path, L: dict[str, str]) -> Chapter:
    m = re.match(r"(\d+)-(.+)\.md$", path.name)
    if not m:
        raise BuildError(f"{path.name}: chapter file must be named NN-slug.md")
    n, slug = int(m[1]), m[2]
    if n == 0:
        return Chapter(path, n, "preface", "", "P")
    if n >= 90:
        am = re.match(r"appendix-([a-z])\b", slug)
        letter = am[1].upper() if am else ""
        return Chapter(path, n, "appendix", f"{L['appendix']} {letter}" if letter else "", letter or "B")
    return Chapter(path, n, "chapter", f"{L['chapter']} {n}", str(n))


def load_chapters(src: Path, lang: str, md: MarkdownIt | None = None) -> list[Chapter]:
    """Parse every NN-slug.md under `src` (filename order). Tokens are untouched by figure/include processing."""
    L = I18N[lang]
    md = md or make_md(L)
    chapters = []
    for p in sorted(src.glob("[0-9][0-9]-*.md")):
        ch = classify(p, L)
        ch.tokens = md.parse(p.read_text())
        h1s = [i for i, t in enumerate(ch.tokens) if t.type == "heading_open" and t.tag == "h1"]
        if len(h1s) != 1:
            ch.errors.append(f"expected exactly one `# ` title, found {len(h1s)}")
        if h1s:
            ch.title = ch.tokens[h1s[0] + 1].content
        chapters.append(ch)
    return chapters


def slugify_fig(n: int) -> str:
    return str(n)


def render_chapter(ch: Chapter, md: MarkdownIt, L: dict[str, str], toc: list) -> str:
    """Transform tokens (headings, figures, includes) and render to HTML. Raises BuildError."""
    tokens = ch.tokens
    cid = f"c{ch.index}"
    if ch.errors:
        raise BuildError(f"{ch.path.name}: {ch.errors[0]}")

    def fail(what: str, msg: str):
        raise BuildError(f"{ch.path.name}: {what}: {msg}")

    # headings: ids, chapter label, toc
    sec = 0
    toc_entry = None
    for i, t in enumerate(tokens):
        if t.type != "heading_open" or t.tag not in ("h1", "h2"):
            continue
        inline = tokens[i + 1]
        if t.tag == "h1":
            t.attrSet("id", cid)
            title = inline.content
            if ch.label and not title.startswith(ch.label):
                prefix = Token("text", "", 0)
                prefix.content = f"{ch.label} — "
                inline.children = [prefix] + (inline.children or [])
                title = f"{ch.label} — {title}"
            ch.title = title
            toc_entry = {"id": cid, "title": title, "subs": []}
            toc.append(toc_entry)
        else:
            sec += 1
            sid = f"{cid}-s{sec}"
            t.attrSet("id", sid)
            if toc_entry is not None:
                toc_entry["subs"].append({"id": sid, "title": inline.content})

    # figures: image-only paragraphs, mermaid + caption line
    fig = 0
    out: list[Token] = []
    i = 0

    def figure_token(html_text: str) -> Token:
        t = Token("html_block", "", 0)
        t.content = html_text
        return t

    def caption_html(text: str) -> str:
        nonlocal fig
        fig += 1
        return (f'<figcaption><span class="fig-no">{L["figure"]} {ch.fig_prefix}.{fig}:</span> '
                f"{md.renderInline(text)}</figcaption>")

    while i < len(tokens):
        t = tokens[i]
        if (t.type == "paragraph_open" and i + 2 < len(tokens)
                and tokens[i + 1].type == "inline" and tokens[i + 1].children
                and len(tokens[i + 1].children) == 1 and tokens[i + 1].children[0].type == "image"):
            img = tokens[i + 1].children[0]
            src = img.attrGet("src") or ""
            if not re.match(r"[a-z]+://", src):
                p = (ch.path.parent / src).resolve()
                if not p.is_file():
                    fail(f"image {src}", "file not found")
                src = p.as_uri()
            out.append(figure_token(
                f'<figure><img src="{html.escape(src)}" alt="{html.escape(img.content)}">'
                f"{caption_html(img.content)}</figure>\n"))
            i += 3
            continue
        if t.type == "fence" and t.info.strip() == "mermaid":
            cap = None
            if (i + 3 < len(tokens) and tokens[i + 1].type == "paragraph_open"
                    and tokens[i + 2].type == "inline"):
                m = CAPTION_RE.match(tokens[i + 2].content.strip())
                if m:
                    cap = m[1]
            body = f'<pre class="mermaid">{html.escape(t.content)}</pre>'
            if cap is not None:
                out.append(figure_token(
                    f'<figure class="diagram">{body}{caption_html(cap)}</figure>\n'))
                i += 4
            else:
                out.append(figure_token(f'<figure class="diagram">{body}</figure>\n'))
                i += 1
            continue
        if t.type == "fence":
            info = t.info.strip().split()[0] if t.info.strip() else ""
            if info == "include":
                try:
                    code, lang, label = resolve_include(t.content)
                except (ValueError, SyntaxError) as e:
                    fail(f"include ({t.content.strip().splitlines()[0] if t.content.strip() else ''})", str(e))
                out.append(figure_token(
                    f'<div class="code-block"><div class="code-label">{html.escape(label)}</div>'
                    f"{highlight_code(code, lang)}</div>\n"))
                i += 1
                continue
            if info == "output":
                try:
                    text = resolve_output(t.content)
                except ValueError as e:
                    fail(f"output ({t.content.strip().splitlines()[0] if t.content.strip() else ''})", str(e))
                out.append(figure_token(
                    f'<div class="output-block"><div class="output-label">{html.escape(L["output"])}</div>'
                    f"<pre>{html.escape(text)}</pre></div>\n"))
                i += 1
                continue
            if info:
                out.append(figure_token(
                    f'<div class="code-block">{highlight_code(t.content.rstrip(chr(10)), info)}</div>\n'))
                i += 1
                continue
        out.append(t)
        i += 1
    html_out = md.renderer.render(out, md.options, {})
    html_out = re.sub(r"<p>(<strong>Q\d+\.\d+</strong>)", r'<p class="q">\1', html_out)
    html_out = re.sub(r"<li>(<strong>Q\d+\.\d+</strong>)", r'<li class="q">\1', html_out)
    return html_out


# --------------------------------------------------------------------------- assembly
def inline_assets() -> tuple[str, str, str]:
    nm = BOOK / "node_modules"
    katex_dist = nm / "katex" / "dist"
    css = (katex_dist / "katex.min.css").read_text()
    fonts = (katex_dist / "fonts").as_uri()
    css = re.sub(r"url\((['\"]?)fonts/", lambda m: f"url({m[1]}{fonts}/", css)
    js = ""
    for rel in ("mermaid/dist/mermaid.min.js", "katex/dist/katex.min.js",
                "katex/dist/contrib/auto-render.min.js"):
        p = nm / rel
        if not p.is_file():
            raise BuildError(f"missing {p} (run `npm i` in book/)")
        js += p.read_text().replace("</script", "<\\/script") + "\n;\n"
    return css, js, ""


PAGE_SCRIPT = """
document.addEventListener('DOMContentLoaded', async () => {
  window.__bookErrors = [];
  try {
    mermaid.initialize({startOnLoad: false, theme: 'neutral', securityLevel: 'loose',
                        flowchart: {useMaxWidth: true, htmlLabels: true}});
    const nodes = Array.from(document.querySelectorAll('pre.mermaid'));
    for (const n of nodes) {
      try { await mermaid.run({nodes: [n], suppressErrors: false}); }
      catch (e) { window.__bookErrors.push('mermaid: ' + (e && e.message || e) + ' :: ' + n.textContent.slice(0, 80)); }
    }
  } catch (e) { window.__bookErrors.push('mermaid init: ' + e); }
  try {
    renderMathInElement(document.body, {
      delimiters: [{left: '\\\\[', right: '\\\\]', display: true},
                   {left: '\\\\(', right: '\\\\)', display: false}],
      throwOnError: false, errorCallback: (m) => window.__bookErrors.push('katex: ' + m)});
  } catch (e) { window.__bookErrors.push('katex: ' + e); }
  if (document.fonts && document.fonts.ready) { await document.fonts.ready; }
  window.__bookReady = true;
});
"""


def assemble(lang: str, src: Path) -> str:
    L = I18N[lang]
    md = make_md(L)
    chapters = load_chapters(src, lang, md)
    if not chapters:
        raise BuildError(f"no chapter files found in {src}")
    toc: list[dict] = []
    bodies = []
    for ch in chapters:
        body = render_chapter(ch, md, L, toc)
        bodies.append(f'<section class="chapter {ch.kind}">\n{body}</section>')
    toc_html = ['<nav class="toc"><h1 class="toc-title">' + html.escape(L["toc"]) + "</h1><ul>"]
    for e in toc:
        toc_html.append(f'<li class="toc1"><a href="#{e["id"]}">{md.renderInline(e["title"])}</a>')
        if e["subs"]:
            toc_html.append("<ul>")
            for s in e["subs"]:
                toc_html.append(f'<li class="toc2"><a href="#{s["id"]}">{md.renderInline(s["title"])}</a></li>')
            toc_html.append("</ul>")
        toc_html.append("</li>")
    toc_html.append("</ul></nav>")
    today = dt.date.today().strftime("%d %B %Y" if lang == "en" else "%d-%m-%Y")
    cover = (f'<section class="cover"><h1 class="cover-title">{html.escape(L["title"])}</h1>'
             f'<p class="cover-sub">{html.escape(L["subtitle"])}</p>'
             f'<p class="cover-date">{L["built"]} {today}</p></section>')
    formatter_css = HtmlFormatter(style="friendly").get_style_defs(".highlight")
    katex_css, js, _ = inline_assets()
    css = (TOOLS / "book.css").read_text()
    return (f'<!doctype html><html lang="{L["lang"]}"><head><meta charset="utf-8">'
            f"<title>{html.escape(L['title'])}</title>"
            f"<style>{katex_css}</style><style>{formatter_css}</style><style>{css}</style></head>"
            f"<body>{cover}{''.join(toc_html)}{''.join(bodies)}"
            f"<script>{js}</script><script>{PAGE_SCRIPT}</script></body></html>")


CONTENT_WIDTH_PX = round((210 - 2 * 18) / 25.4 * 96)  # A4 text width in CSS px
OVERFLOW_JS = """(limit) => {
  const out = [];
  for (const el of document.querySelectorAll('section.chapter > *, section > *')) {
    if (el.scrollWidth > limit + 4 && !el.closest('.katex-mathml')) {
      const h = el.closest('section')?.querySelector('h1')?.textContent ?? '?';
      out.push(`${h}: <${el.tagName.toLowerCase()}> ${el.scrollWidth}px: ${(el.textContent || '').trim().slice(0, 80)}`);
    }
  }
  return out.slice(0, 10);
}"""


def build_pdf(html_path: Path, pdf_path: Path) -> int:
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        # Viewport = A4 width minus the 18 mm side margins, so the overflow check below sees print layout.
        page = browser.new_page(viewport={"width": CONTENT_WIDTH_PX, "height": 1000})
        page.on("pageerror", lambda e: print(f"  [page error] {e}", file=sys.stderr))
        page.goto(html_path.resolve().as_uri())
        page.wait_for_function("window.__bookReady === true", timeout=120000)
        errors = page.evaluate("window.__bookErrors")
        if errors:
            browser.close()
            raise BuildError("render errors in page:\n  " + "\n  ".join(errors))
        overflow = page.evaluate(OVERFLOW_JS, CONTENT_WIDTH_PX)
        if overflow:
            browser.close()
            # Chromium would silently shrink EVERY page to fit the widest element.
            raise BuildError("content wider than the page (usually a glued ``` fence or an indented "
                             "paragraph turned into a code block):\n  " + "\n  ".join(overflow))
        page.pdf(
            path=str(pdf_path), format="A4", print_background=True,
            display_header_footer=True,
            margin={"top": "18mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
            header_template="<span></span>",
            footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#666">'
                            '<span class="pageNumber"></span></div>')
        browser.close()
    return len(PdfReader(str(pdf_path)).pages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "id", "all"], default="all")
    ap.add_argument("--src", type=Path, help="chapter directory override (single --lang only)")
    ap.add_argument("--out", type=Path, help="PDF path override (single --lang only)")
    ap.add_argument("--html-only", action="store_true", help="skip the PDF step")
    args = ap.parse_args()
    langs = ["en", "id"] if args.lang == "all" else [args.lang]
    if (args.src or args.out) and len(langs) != 1:
        ap.error("--src/--out need a single --lang")
    (BOOK / "build").mkdir(exist_ok=True)
    for lang in langs:
        src = args.src or BOOK / lang
        try:
            doc = assemble(lang, src)
            # With --out, keep the HTML next to that PDF so concurrent test builds don't overwrite each other.
            html_path = (args.out.with_suffix(".html") if args.out else BOOK / "build" / f"book-{lang}.html").resolve()
            html_path.write_text(doc)
            print(f"[{lang}] wrote {html_path.relative_to(ROOT)} ({len(doc) // 1024} KiB)")
            if args.html_only:
                continue
            pdf = args.out or BOOK / "dist" / f"recsys-book-{lang}.pdf"
            pages = build_pdf(html_path, pdf)
            print(f"[{lang}] wrote {pdf} — {pages} pages")
        except BuildError as e:
            print(f"[{lang}] BUILD ERROR: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
