"""一键运行数据整理、切块、建索引与评测。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_command(command: list[str]) -> None:
    print("\n>>>", " ".join(command))
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    python_bin = sys.executable
    run_command([python_bin, "prepare_corpus.py"])
    run_command([python_bin, "chunk_docs.py", "--chunk_size", str(args.chunk_size), "--overlap", str(args.overlap)])
    run_command([python_bin, "build_index.py", "--backend", args.backend])
    run_command([python_bin, "evaluate_retrieval.py", "--backend", args.backend])
    run_command([python_bin, "evaluate_answers.py", "--backend", args.backend])
    print("\n主流程已完成。现在可以运行：")
    print(f"{python_bin} retrieve_demo.py --query \"什么是压缩映射原理？\" --backend {args.backend} --topk 3")
    print(f"{python_bin} generate_answer_rule.py --query \"什么是压缩映射原理？\" --backend {args.backend} --topk 3")
    print(f"{python_bin} app.py")


if __name__ == "__main__":
    main()
