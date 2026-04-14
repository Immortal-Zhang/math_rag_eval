# 文件说明

## 核心脚本

- `prepare_corpus.py`  
  读取 `data/raw/source_txt/` 中的原始文本资料，整理成统一的 `docs.jsonl`。

- `chunk_docs.py`  
  把 `docs.jsonl` 切分成带重叠的文本块，输出到 `data/processed/chunks.jsonl`。

- `build_index.py`  
  根据 `chunks.jsonl` 构建检索索引。支持 `tfidf` 和可选的 `sbert`。

- `retrieve_demo.py`  
  输入一个问题，查看 top-k 检索结果，可保存成文本文件，适合做演示样例。

- `generate_answer_rule.py`  
  在检索结果基础上生成规则式回答，是项目的 baseline 回答脚本。

- `generate_answer_api.py`  
  在检索证据基础上调用 OpenAI 兼容接口生成回答。没有密钥时可以先不用。

- `evaluate_retrieval.py`  
  读取评测问题集，计算 `Recall@1 / Recall@3 / Recall@5`。

- `evaluate_answers.py`  
  生成规则式回答，并输出关键词覆盖率和人工打分模板。

- `run_pipeline.py`  
  一键串起数据整理、切块、建索引与评测。

- `app.py`  
  启动 `Gradio` 页面，用于项目展示。

## 工具文件

- `rag_utils.py`  
  检索相关公共函数，包括 `jsonl` 读写、切块、建索引、检索、指标函数等。

- `answer_utils.py`  
  回答生成与评测相关公共函数，包括句子切分、证据句抽取、规则式回答、关键词覆盖率等。

## 数据与结果

- `data/raw/source_txt/`  
  原始文本资料目录。当前附带了一份示例语料。

- `data/raw/eval_queries.jsonl`  
  评测问题集，每条包括 `query_id`、`query`、`relevant_doc_id` 和 `reference_keywords`。

- `data/raw/docs.jsonl`  
  由 `prepare_corpus.py` 生成的统一文档格式文件。

- `data/processed/chunks.jsonl`  
  由 `chunk_docs.py` 生成的文本块文件。

- `artifacts/`  
  检索索引、评测结果、示例检索输出、示例回答输出等都会保存在这里。

## 说明文档

- `README.md`  
  项目主页，给别人看的第一份文档。

- `docs/project_report.md`  
  项目报告模板，建议你把自己的数据规模、结果和 bad case 都写进去。

- `.gitignore`  
  Git 提交忽略规则，避免把缓存和大文件一起传上去。
