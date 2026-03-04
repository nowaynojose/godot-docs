import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from bs4 import BeautifulSoup


ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_ROOT = ROOT_DIR / "_build" / "html"
OUTPUT_PATH = ROOT_DIR / "_build" / "godot_rag_dataset.jsonl"


EXCLUDED_BASENAMES = {
    "genindex.html",
    "search.html",
    "objects.inv",
}


def iter_html_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise SystemExit(f"HTML build directory not found: {root}")

    for path in root.rglob("*.html"):
        if path.name in EXCLUDED_BASENAMES:
            continue
        yield path


def extract_page(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    body = soup.body or soup

    # Prefer first <h1> for title, fall back to <title>.
    title_tag = body.find("h1")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    else:
        title = path.stem

    text = body.get_text(separator=" ", strip=True)

    rel_path = path.relative_to(ROOT_DIR).as_posix()
    return {
        "source": rel_path,
        "title": title,
        "content": text,
    }


def chunk_words(words: List[str], chunk_size: int, overlap: int) -> Iterable[List[str]]:
    if not words:
        return

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        yield words[start:end]

        if end == n:
            break

        start = end - overlap


def iter_chunked_records(
    page: Dict[str, str],
    chunk_size: int = 600,
    overlap: int = 100,
) -> Iterable[Dict[str, str]]:
    words = page["content"].split()
    chunk_index = 0

    for chunk_words_list in chunk_words(words, chunk_size, overlap):
        if not chunk_words_list:
            continue

        chunk_text = " ".join(chunk_words_list)
        record = {
            "source": page["source"],
            "title": page["title"],
            "content": chunk_text,
            "chunk_index": chunk_index,
        }
        yield record
        chunk_index += 1


def build_dataset(
    html_root: Path = HTML_ROOT,
    output_path: Path = OUTPUT_PATH,
    chunk_size: int = 600,
    overlap: int = 100,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for html_file in iter_html_files(html_root):
            page = extract_page(html_file)
            total_pages += 1

            for record in iter_chunked_records(page, chunk_size=chunk_size, overlap=overlap):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Wrote {total_chunks} chunks from {total_pages} HTML pages to {output_path}")


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup


@dataclass
class PageRecord:
    source: str
    title: str
    content: str


@dataclass
class ChunkRecord:
    source: str
    title: str
    content: str
    chunk_id: str
    chunk_index: int


SKIP_BASENAMES = {
    "search.html",
    "genindex.html",
    "searchindex.html",
    "objects.inv",
}


def iter_html_files(html_root: Path) -> Iterable[Path]:
    for path in html_root.rglob("*.html"):
        if path.name in SKIP_BASENAMES:
            continue
        yield path


def extract_page(path: Path, docs_root: Path) -> PageRecord:
    # Use UTF-8 with errors ignored to be resilient to odd encodings
    text = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "lxml")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(separator=" ", strip=True) if title_tag else path.stem

    # Remove obvious navigation sections if they exist
    for nav_selector in ("div.related", "div.sphinxsidebar", "nav", "footer"):
        for node in soup.select(nav_selector):
            node.decompose()

    content = soup.get_text(separator=" ", strip=True)

    rel_source = path.relative_to(docs_root).as_posix()
    return PageRecord(source=rel_source, title=title, content=content)


def chunk_text(
    text: str,
    *,
    target_words: int = 600,
    overlap_words: int = 100,
) -> List[str]:
    """
    Approximate 400–800 tokens using word counts.

    With common English, ~1.3–1.5 tokens per word is a decent rule of thumb,
    so 600 words is a reasonable middle point for 400–800 tokens.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(target_words - overlap_words, 1)

    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step

    return chunks


def page_to_chunks(page: PageRecord) -> List[ChunkRecord]:
    chunks = chunk_text(page.content)
    records: List[ChunkRecord] = []

    # Derive a stable prefix for chunk_id from the source path
    base = page.source.replace("/", "_").replace("\\", "_")

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{base}__{idx}"
        title = page.title
        if len(chunks) > 1:
            title = f"{page.title} (part {idx + 1})"

        records.append(
            ChunkRecord(
                source=page.source,
                title=title,
                content=chunk,
                chunk_id=chunk_id,
                chunk_index=idx,
            )
        )

    return records


def write_chunks_jsonl(chunks: Iterable[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in chunks:
            obj = asdict(record)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_dataset(
    docs_root: Path | None = None,
    html_subdir: str = "_build/html",
    output_rel_path: str = "_build/godot_rag_dataset.jsonl",
) -> Path:
    if docs_root is None:
        docs_root = Path(__file__).resolve().parents[1]

    html_root = docs_root / html_subdir
    if not html_root.exists():
        raise FileNotFoundError(
            f"HTML root not found at {html_root}. "
            "Make sure you have built the docs with Sphinx first."
        )

    output_path = docs_root / output_rel_path

    all_chunks: List[ChunkRecord] = []
    for html_file in iter_html_files(html_root):
        page = extract_page(html_file, docs_root)
        page_chunks = page_to_chunks(page)
        all_chunks.extend(page_chunks)

    write_chunks_jsonl(all_chunks, output_path)
    return output_path


def main() -> None:
    docs_root_env = os.getenv("GODOT_DOCS_ROOT")
    docs_root = Path(docs_root_env) if docs_root_env else None

    output_path = build_dataset(docs_root=docs_root)
    print(f"Wrote RAG dataset to: {output_path}")


if __name__ == "__main__":
    main()

