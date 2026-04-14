"""演示单条 query 的检索结果。"""

from __future__ import annotations

import argparse

from rag_utils import build_index_path, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()

    index_path = build_index_path(args.backend)
    results = retrieve(args.query, index_path=index_path, topk=args.topk)

    lines = [
        f"查询：{args.query}",
        f"后端：{args.backend}",
        f"TopK：{args.topk}",
        "-" * 60,
    ]
    for rank, item in enumerate(results, start=1):
        lines.append(
            f"[{rank}] doc_id={item['doc_id']} | title={item['title']} | score={item['score']:.4f}"
        )
        lines.append(item["text"])
        lines.append("-" * 60)

    output_text = "\n".join(lines)
    print(output_text)

    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as handle:
            handle.write(output_text)
        print(f"已保存到：{args.save_path}")


if __name__ == "__main__":
    main()
