"""将 docs.jsonl 切分为带重叠的文本块。"""

from __future__ import annotations

import argparse

from rag_utils import PROCESSED_DIR, RAW_DIR, chunk_text, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    documents = read_jsonl(RAW_DIR / "docs.jsonl")
    if not documents:
        raise FileNotFoundError("没有找到 docs.jsonl，请先运行 prepare_corpus.py。")

    chunk_records: list[dict[str, str]] = []
    for document in documents:
        chunks = chunk_text(
            document["text"],
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        for idx, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{document['doc_id']}_chunk_{idx:02d}",
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "text": chunk,
                }
            )

    output_path = PROCESSED_DIR / "chunks.jsonl"
    write_jsonl(output_path, chunk_records)

    print(f"共生成 {len(chunk_records)} 个 chunk。")
    print(f"输出路径：{output_path}")


if __name__ == "__main__":
    main()
