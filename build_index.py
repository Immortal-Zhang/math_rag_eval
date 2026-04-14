"""根据 chunks.jsonl 构建检索索引。"""

from __future__ import annotations

from rag_utils import (
    PROCESSED_DIR,
    build_index_path,
    build_sbert_index,
    build_tfidf_index,
    parse_backend_args,
    read_jsonl,
)


def main() -> None:
    parser = parse_backend_args()
    args = parser.parse_args()

    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    if not chunks:
        raise FileNotFoundError("没有找到 chunks.jsonl，请先运行 chunk_docs.py。")

    index_path = build_index_path(args.backend)
    if args.backend == "tfidf":
        build_tfidf_index(chunks, index_path)
    elif args.backend == "sbert":
        build_sbert_index(chunks, index_path)
    else:
        raise ValueError(f"未知 backend：{args.backend}")

    print(f"索引已保存到：{index_path}")


if __name__ == "__main__":
    main()
