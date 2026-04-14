"""检索增强项目的公共工具函数。"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SOURCE_DIR = RAW_DIR / "source_txt"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for path in [SOURCE_DIR, PROCESSED_DIR, ARTIFACTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 jsonl 文件。若文件不存在，返回空列表。"""
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    """写入 jsonl 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    """做最小清洗，合并多余空白字符。"""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_source_filename(path: Path) -> tuple[str, str]:
    """
    从文件名解析 doc_id 和 title。
    推荐格式：doc_01_压缩映射原理.txt
    """
    stem = path.stem
    match = re.match(r"^(doc_\d+?)_(.+)$", stem)
    if match:
        return match.group(1), match.group(2)
    return stem, stem


def load_source_documents() -> list[dict[str, str]]:
    """从 source_txt 目录读取原始语料，整理成统一结构。"""
    documents: list[dict[str, str]] = []
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        doc_id, title = parse_source_filename(path)
        text = clean_text(path.read_text(encoding="utf-8"))
        if not text:
            continue
        documents.append({"doc_id": doc_id, "title": title, "text": text})
    return documents


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """
    对文本做重叠切块。
    overlap 的作用是减少定义句、公式说明句被截断后语义丢失。
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size。")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def compose_retrieval_text(item: dict[str, Any]) -> str:
    """把标题和正文拼在一起，并对标题做轻量加权。"""
    title = str(item.get("title", "")).strip()
    body = str(item.get("text", "")).strip()
    return f"{title} {title} {body}".strip()


def build_tfidf_index(chunks: list[dict[str, Any]], save_path: Path) -> None:
    """构建 TF-IDF 字符 n-gram 检索索引。"""
    texts = [compose_retrieval_text(item) for item in chunks]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(texts)
    payload = {
        "backend": "tfidf",
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": chunks,
    }
    with open(save_path, "wb") as handle:
        pickle.dump(payload, handle)


def build_sbert_index(chunks: list[dict[str, Any]], save_path: Path) -> None:
    """构建 SentenceTransformer 向量索引。"""
    from sentence_transformers import SentenceTransformer

    texts = [compose_retrieval_text(item) for item in chunks]
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    payload = {
        "backend": "sbert",
        "model_name": model_name,
        "embeddings": embeddings,
        "chunks": chunks,
    }
    with open(save_path, "wb") as handle:
        pickle.dump(payload, handle)


def build_index_path(backend: str) -> Path:
    return ARTIFACTS_DIR / f"index_{backend}.pkl"


def load_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(f"没有找到索引文件：{index_path}")
    with open(index_path, "rb") as handle:
        return pickle.load(handle)


def _scores_to_numpy(scores: Any) -> np.ndarray:
    if hasattr(scores, "toarray"):
        return scores.toarray().reshape(-1)
    return np.asarray(scores).reshape(-1)


def retrieve(query: str, index_path: Path, topk: int = 5) -> list[dict[str, Any]]:
    """根据 query 返回 top-k 结果。"""
    payload = load_index(index_path)
    chunks = payload["chunks"]

    if payload["backend"] == "tfidf":
        vectorizer: TfidfVectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
        query_vec = vectorizer.transform([query])
        scores_all = _scores_to_numpy(query_vec @ matrix.T)
    elif payload["backend"] == "sbert":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(payload["model_name"])
        query_vec = model.encode([query], normalize_embeddings=True)
        doc_matrix = payload["embeddings"]
        scores_all = np.asarray(query_vec @ doc_matrix.T).reshape(-1)
    else:
        raise ValueError(f"暂不支持的 backend：{payload['backend']}")

    indices = np.argsort(scores_all)[::-1][:topk]
    results: list[dict[str, Any]] = []
    for idx in indices:
        item = dict(chunks[int(idx)])
        item["score"] = float(scores_all[int(idx)])
        results.append(item)
    return results


def extract_keywords(query: str) -> list[str]:
    """从中文问题中做一个轻量关键词抽取。"""
    cleaned = query.strip()
    cleaned = re.sub(r"[？?]", "", cleaned)

    stop_phrases = [
        "为什么通常比",
        "适合解决什么问题",
        "主要计算代价来自哪里",
        "相比",
        "有什么用",
        "什么时候",
        "为什么",
        "什么是",
        "是什么",
        "什么意思",
        "是什么意思",
        "有哪些",
        "区别",
        "核心思想",
    ]
    for phrase in sorted(stop_phrases, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ")

    segments = [
        segment.strip(" 的了呢吗嘛啊呀与和在对比会能")
        for segment in re.split(r"[，。！？；：、（）()【】\[\]{}<>“”‘’\s]+", cleaned)
        if segment.strip()
    ]

    candidates: list[str] = []
    for segment in segments:
        if len(segment) >= 2:
            candidates.append(segment)
        max_n = min(6, len(segment))
        for n in range(max_n, 1, -1):
            for start in range(0, len(segment) - n + 1):
                piece = segment[start:start + n]
                if len(piece.strip()) >= 2:
                    candidates.append(piece)

    seen = set()
    keywords: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            keywords.append(item)
    return keywords


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_id: str, k: int) -> int:
    return int(relevant_doc_id in retrieved_doc_ids[:k])


def parse_backend_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="tfidf", choices=["tfidf", "sbert"])
    parser.add_argument("--topk", type=int, default=5)
    return parser
