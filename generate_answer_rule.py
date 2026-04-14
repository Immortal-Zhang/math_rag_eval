"""基于检索结果生成规则式回答。"""

from __future__ import annotations

import argparse

from answer_utils import generate_rule_answer
from rag_utils import build_index_path, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="")
    args = parser.parse_args()

    index_path = build_index_path(args.backend)
    retrieved = retrieve(args.query, index_path=index_path, topk=args.topk)
    answer = generate_rule_answer(args.query, retrieved)
    print(answer)

    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as handle:
            handle.write(answer)
        print(f"已保存到：{args.save_path}")


if __name__ == "__main__":
    main()
