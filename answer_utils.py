"""回答生成与简单评测相关函数。"""

from __future__ import annotations

import re
from typing import Iterable

from rag_utils import extract_keywords


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；]", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_match_score(sentence: str, keywords: list[str]) -> tuple[int, int]:
    keyword_hits = sum(1 for keyword in keywords if keyword in sentence)
    char_hits = sum(1 for char in set("".join(keywords)) if char and char in sentence)
    return keyword_hits, char_hits


def select_evidence_sentences(query: str, retrieved: list[dict], max_sentences: int = 4) -> list[str]:
    """优先抽取包含查询关键词的句子，并按匹配强度排序。"""
    keywords = extract_keywords(query)
    scored_sentences: list[tuple[tuple[int, int, int], str]] = []

    for item_rank, item in enumerate(retrieved):
        for sentence_rank, sentence in enumerate(split_sentences(item["text"])):
            hit_score, char_score = sentence_match_score(sentence, keywords)
            priority = (hit_score, char_score, -item_rank * 10 - sentence_rank)
            scored_sentences.append((priority, sentence))

    scored_sentences.sort(reverse=True)

    selected: list[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        if sentence not in seen:
            seen.add(sentence)
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    return selected


def build_rule_summary(evidence_sentences: list[str]) -> str:
    if not evidence_sentences:
        return "当前检索到的资料不足，建议补充更相关语料后再回答。"
    if len(evidence_sentences) == 1:
        return evidence_sentences[0] + "。"
    return "；".join(evidence_sentences[:2]) + "。"


def generate_rule_answer(query: str, retrieved: list[dict]) -> str:
    """基于检索到的证据做规则式回答。"""
    evidence_sentences = select_evidence_sentences(query, retrieved)
    summary = build_rule_summary(evidence_sentences)
    evidence_block = "\n".join(f"- {sentence}" for sentence in evidence_sentences) or "- 暂无有效证据"

    return (
        f"问题：{query}\n\n"
        f"规则式回答：\n{summary}\n\n"
        f"证据片段：\n{evidence_block}\n\n"
        f"说明：该版本使用检索结果中的句子抽取与重组，不依赖外部大模型接口。"
    )


def build_context_text(retrieved: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(retrieved, start=1):
        blocks.append(
            f"[资料 {idx}] 标题：{item['title']}\n"
            f"文档编号：{item['doc_id']}\n"
            f"内容：{item['text']}"
        )
    return "\n\n".join(blocks)


def build_api_prompt(query: str, retrieved: list[dict]) -> str:
    context = build_context_text(retrieved)
    return f"""你是一个严谨的数学助教，请严格依据给定资料回答问题。
如果资料不足，请明确指出“资料不足”，不要编造。

用户问题：
{query}

给定资料：
{context}

请按下面结构回答：
1. 简明结论
2. 关键依据
3. 需要提醒的条件或边界
"""


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hit = sum(1 for keyword in keywords if keyword in answer)
    return hit / len(keywords)
