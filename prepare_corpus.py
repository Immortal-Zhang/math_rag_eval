"""读取 source_txt 目录中的文本资料，整理为 docs.jsonl。"""

from __future__ import annotations

from rag_utils import RAW_DIR, SOURCE_DIR, load_source_documents, read_jsonl, write_jsonl


def main() -> None:
    documents = load_source_documents()
    if not documents:
        raise FileNotFoundError(
            f"没有在 {SOURCE_DIR} 找到原始资料。请先放入若干 .txt 文件。"
        )

    write_jsonl(RAW_DIR / "docs.jsonl", documents)

    eval_queries = read_jsonl(RAW_DIR / "eval_queries.jsonl")
    print(f"已整理 {len(documents)} 篇文档到 {RAW_DIR / 'docs.jsonl'}。")
    if eval_queries:
        print(f"检测到 {len(eval_queries)} 条评测问题：{RAW_DIR / 'eval_queries.jsonl'}")
    else:
        print("尚未检测到 eval_queries.jsonl，后续评测脚本会无法运行。")


if __name__ == "__main__":
    main()
