"""生成规则式回答，并输出自动评测与人工打分模板。"""

from __future__ import annotations

import argparse
import csv
import json

from answer_utils import generate_rule_answer, keyword_coverage
from rag_utils import ARTIFACTS_DIR, RAW_DIR, build_index_path, read_jsonl, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    queries = read_jsonl(RAW_DIR / "eval_queries.jsonl")
    if not queries:
        raise FileNotFoundError("没有找到 eval_queries.jsonl。")

    index_path = build_index_path(args.backend)
    rows = []
    coverages: list[float] = []

    for item in queries:
        retrieved = retrieve(item["query"], index_path=index_path, topk=args.topk)
        answer = generate_rule_answer(item["query"], retrieved)
        coverage = keyword_coverage(answer, item["reference_keywords"])
        coverages.append(coverage)

        rows.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "answer": answer,
                "keyword_coverage": round(coverage, 4),
                "manual_score_0_2": "",
                "manual_comment": "",
            }
        )

    average_coverage = round(sum(coverages) / len(coverages) if coverages else 0.0, 4)
    csv_path = ARTIFACTS_DIR / f"answer_eval_{args.backend}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "answer",
                "keyword_coverage",
                "manual_score_0_2",
                "manual_comment",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = ARTIFACTS_DIR / f"answer_summary_{args.backend}.json"
    json_path.write_text(
        json.dumps({"average_keyword_coverage": average_coverage}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"平均关键词覆盖率：{average_coverage:.4f}")
    print(f"自动评测与人工打分模板已保存到：{csv_path}")
    print(f"汇总结果已保存到：{json_path}")
    print("人工评分建议：2=正确且完整；1=部分正确；0=明显错误或幻觉。")


if __name__ == "__main__":
    main()
