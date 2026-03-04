"""
Build a FAISS vector index from godot_rag_dataset.jsonl for semantic search.

Requires: sentence-transformers, faiss-cpu
    pip install sentence-transformers faiss-cpu

Outputs to _build/:
  - godot_rag_index.faiss   (vector index)
  - godot_rag_metadata.json (chunk metadata for lookups)
"""

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "_build" / "godot_rag_dataset.jsonl"
INDEX_PATH = ROOT_DIR / "_build" / "godot_rag_index.faiss"
METADATA_PATH = ROOT_DIR / "_build" / "godot_rag_metadata.json"

# Lightweight model, good balance of speed and quality
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_index(
    dataset_path: Path = DATASET_PATH,
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run tools/extract_html_docs.py first."
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Run: pip install sentence-transformers"
        )
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu is required. Run: pip install faiss-cpu")

    chunks = load_chunks(dataset_path)
    if not chunks:
        raise ValueError(f"No chunks found in {dataset_path}")

    texts = [c["content"] for c in chunks]
    metadata = [
        {
            "source": c.get("source", ""),
            "title": c.get("title", ""),
            "chunk_index": c.get("chunk_index", i),
            "chunk_id": c.get("chunk_id", f"chunk_{i}"),
        }
        for i, c in enumerate(chunks)
    ]

    print(f"Embedding {len(chunks)} chunks with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vecs)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dimension": dimension,
                "num_chunks": len(chunks),
                "chunks": metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote index to {index_path}")
    print(f"Wrote metadata to {metadata_path}")


def main() -> None:
    docs_root = os.getenv("GODOT_DOCS_ROOT")
    root = Path(docs_root) if docs_root else ROOT_DIR

    build_index(
        dataset_path=root / "_build" / "godot_rag_dataset.jsonl",
        index_path=root / "_build" / "godot_rag_index.faiss",
        metadata_path=root / "_build" / "godot_rag_metadata.json",
    )


if __name__ == "__main__":
    main()
