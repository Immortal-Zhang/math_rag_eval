"""计算检索 Recall@1 / Recall@3 / Recall@5，并输出逐条结果。"""

from __future__ import annotations

import argparse
import csv
import json

from rag_utils import ARTIFACTS_DIR, RAW_DIR, build_index_path, read_jsonl, recall_at_k, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    args = parser.parse_args()

    queries = read_jsonl(RAW_DIR / "eval_queries.jsonl")
    if not queries:
        raise FileNotFoundError("没有找到 eval_queries.jsonl。")

    index_path = build_index_path(args.backend)
    ks = [1, 3, 5]
    hit_count = {k: 0 for k in ks}
    rows = []

    for item in queries:
        results = retrieve(item["query"], index_path=index_path, topk=max(ks))
        retrieved_doc_ids = [result["doc_id"] for result in results]
        row = {
            "query_id": item["query_id"],
            "query": item["query"],
            "relevant_doc_id": item["relevant_doc_id"],
        }
        for k in ks:
            hit = recall_at_k(retrieved_doc_ids, item["relevant_doc_id"], k)
            hit_count[k] += hit
            row[f"hit@{k}"] = hit
        rows.append(row)

    total = len(queries)
    summary = {
        f"R@{k}": round(hit_count[k] / total if total else 0.0, 4)
        for k in ks
    }

    csv_path = ARTIFACTS_DIR / f"retrieval_eval_{args.backend}.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_id", "query", "relevant_doc_id", "hit@1", "hit@3", "hit@5"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = ARTIFACTS_DIR / f"retrieval_summary_{args.backend}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"评测样本数：{total}")
    for name, value in summary.items():
        print(f"{name} = {value:.4f}")
    print(f"逐条结果已保存到：{csv_path}")
    print(f"汇总结果已保存到：{json_path}")


if __name__ == "__main__":
    main()
