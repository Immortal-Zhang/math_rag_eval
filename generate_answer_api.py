"""调用 OpenAI 兼容接口，在检索证据基础上生成回答。"""

from __future__ import annotations

import argparse
import os

from openai import OpenAI

from answer_utils import build_api_prompt
from rag_utils import build_index_path, retrieve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"没有找到环境变量 {args.api_key_env}。"
            f"若暂时不使用外部接口，可先运行 generate_answer_rule.py。"
        )

    index_path = build_index_path(args.backend)
    retrieved = retrieve(args.query, index_path=index_path, topk=args.topk)
    prompt = build_api_prompt(args.query, retrieved)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "你是严谨的数学助教。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
