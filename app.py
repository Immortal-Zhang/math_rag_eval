"""Gradio 演示页面。"""

from __future__ import annotations

import os

import gradio as gr

from answer_utils import generate_rule_answer
from rag_utils import build_index_path, retrieve


def run_demo(query: str, backend: str, topk: int) -> tuple[str, str]:
    if not query.strip():
        return "请输入一个数学问题。", ""

    index_path = build_index_path(backend)
    if not os.path.exists(index_path):
        return "请先构建索引，再打开演示页面。", ""

    retrieved = retrieve(query, index_path=index_path, topk=topk)
    answer = generate_rule_answer(query, retrieved)

    lines = []
    for rank, item in enumerate(retrieved, start=1):
        lines.append(
            f"### Top {rank}\n"
            f"- 文档：{item['title']} ({item['doc_id']})\n"
            f"- 分数：{item['score']:.4f}\n"
            f"- 内容：{item['text']}\n"
        )
    return answer, "\n".join(lines)


def main() -> None:
    demo = gr.Interface(
        fn=run_demo,
        inputs=[
            gr.Textbox(label="输入数学问题", lines=2, placeholder="例如：什么是压缩映射原理？"),
            gr.Dropdown(choices=["tfidf", "sbert"], value="tfidf", label="检索后端"),
            gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top K"),
        ],
        outputs=[
            gr.Textbox(label="规则式回答"),
            gr.Markdown(label="检索到的证据片段"),
        ],
        title="数学资料检索问答演示",
        description="推荐先运行 run_pipeline.py，再打开本页面演示。",
    )
    demo.launch()


if __name__ == "__main__":
    main()
