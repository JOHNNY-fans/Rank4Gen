#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Ranker evaluation script (prompt building + inference).

Given:
  - a test dataset jsonl (each line contains query + candidate documents)
  - language (en/zh)
  - inference mode (index/snapshot)
  - tokenizer path (used to compute token budget + truncate docs)
  - ranker model name (for chat.completions)
  - api_url (OpenAI-compatible base_url)
  - output_path (jsonl)

This script will:
  1) Load model descriptions from built-in default paths:
       - model_descriptions.json (en)
       - model_descriptions_zh.json (zh)
  2) Build a system prompt with the selected downstream generator model description
  3) Compute the available token budget for <Documents> automatically
  4) Truncate docs to fit the budget (token-accurate truncation)
  5) Call the Ranker API and parse ranked doc indices from the visible output
  6) Write results to output jsonl with resume-by-sample_id

Input dataset format (jsonl), each line:
{
  "id": "...",                 # or "sample_id"
  "query": "...",
  "source": "...",
  "answers": ["..."],
  "documents": [
    {"id": 1, "text": "..."},
    {"id": 2, "text": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoTokenizer
from openai import OpenAI
from tqdm import tqdm

# =========================
# Defaults (edit if needed)
# =========================

DEFAULT_DESC_FILE = {
    "en": "model_descriptions/model_descriptions.json",
    "zh": "model_descriptions/model_descriptions_zh.json",
}

# Context budgeting
RANKER_MAX_CONTEXT_TOKENS = 40960
RANKER_MAX_OUTPUT_TOKENS = 4096
PROMPT_TOKEN_BUDGET = RANKER_MAX_CONTEXT_TOKENS - RANKER_MAX_OUTPUT_TOKENS

# Truncation mark appended when cutting a doc
TRUNCATION_MARK = " <TRUNCATED>"

# ChatML-ish wrappers used for token counting (must match the tokenizer conventions you use)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"

# =========================
# Prompt templates
# =========================

EN_SYSTEM_PROMPT_TEMPLATE = """You are **Rank4Gen**, a **Ranker** designed for retrieval-augmented generation tasks.
Given a **Query (<Query>)** and **Candidate Documents (<Documents>)**, you need to **select and rank** the documents from a set of candidate documents that are most suitable for the downstream generator to answer the query, based on the characteristics and preferences of **Downstream Generator Information**.

When the downstream generator is `default`, it indicates a default mode with no specific preferences. In this case, you should **select and rank** the candidate documents that are **most helpful for the query** and **most directly support answering it**.

Please **strictly follow** the **Instructions (<Instruct>)** below for document selection and ranking.

---

## Downstream Generator Information

The downstream generator you serve is: `{downstream_model}`  
Generator description: `{description}`

---

## Output Mode

### 1. Index Mode
If the instruction contains **`/index`**, output only the **document index**, one per line, without additional text or explanation.

**Example:**
[<doc_index_1>]
[<doc_index_2>]
[<doc_index_3>]

### 2. Snapshot Mode
If the instruction contains **`/snapshot`**, output the selected documents **line by line** using *snapshot format*.  
Each line must include:

- **Document index**  
- **Preview of the first 100 characters** of the document content  

**Example:**
[<doc_index_1>] <first_100_characters_of_document>...
[<doc_index_2>] <first_100_characters_of_document>...
[<doc_index_3>] <first_100_characters_of_document>..."""

EN_USER_PROMPT_TEMPLATE = """<Instruct>: I will provide you with {num} documents, each indicated by a numerical identifier []. Select the documents based on their relevance to the search query "{question}".

<Query>: {question}

<Documents>: 
{context}

Select the documents that mostly cover clear and diverse information to answer the query.

Please output the final document selection and sorting results according to the format constraints of the **"Output Mode"**.

<Output>:"""

ZH_SYSTEM_PROMPT_TEMPLATE = """你是**Rank4Gen**，一个检索增强生成任务的**Ranker**。  
给定**查询 (<Query>)**与**候选文档 (<Documents>)**，你需要根据**下游生成器信息**的特点和偏好，从候选文档中**筛选并排序**出最适合该生成器回答的文档。

当下游生成器为`default`时，代表无偏好的默认模式，你需要从候选文档中**选择并排序**出**对该查询最有帮助**、**最能直接支持回答**的文档。

请**严格按照**下方的**指令 (<Instruct>)**进行文档选择与排序。

---

## 下游生成器信息

你所服务的下游生成器是：`{downstream_model}`
生成器描述：`{description}`

---

## 输出模式

### 1. Index 模式
如果指令中包含 ** `/index`**，则仅输出 **文档索引**，每行一个，不添加任何解释或额外文本。

**示例:**
[<doc_index_1>]
[<doc_index_2>]
[<doc_index_3>]

### 2. Snapshot 模式
如果指令中包含 **`/snapshot`**，请使用 *snapshot 格式* **逐行输出**所选文档。  
每行必须包括：

- **文档索引**  
- **文档内容前 100 个字符的预览**

**示例：**
[<doc_index_1>] <first_100_characters_of_document>...
[<doc_index_2>] <first_100_characters_of_document>...
[<doc_index_3>] <first_100_characters_of_document>..."""

ZH_USER_PROMPT_TEMPLATE = """<Instruct>: 我将向你提供 {num} 个文档，每个文档都有一个数字标识符 []。请根据它们与搜索查询“{question}”的相关性选择段落。

<Query>: {question}

<Documents>:
{context}

请选择那些能够提供清晰且多样信息、最能回答查询的文档。

请根据 “输出模式” 的格式要求输出最终的文档选择和排序结果。

<Output>: """

SYSTEM_TEMPLATES = {"en": EN_SYSTEM_PROMPT_TEMPLATE, "zh": ZH_SYSTEM_PROMPT_TEMPLATE}
USER_TEMPLATES = {"en": EN_USER_PROMPT_TEMPLATE, "zh": ZH_USER_PROMPT_TEMPLATE}


# =========================
# Token helpers
# =========================

def _encode_no_special(tok: AutoTokenizer, s: str) -> List[int]:
    return tok(s, add_special_tokens=False)["input_ids"]


def init_token_constants(tok: AutoTokenizer) -> Dict[str, Any]:
    return {
        "im_end_ids": _encode_no_special(tok, IM_END),
        "im_start_role_ids": {
            "system": _encode_no_special(tok, f"{IM_START}system\n"),
            "user": _encode_no_special(tok, f"{IM_START}user\n"),
            "assistant": _encode_no_special(tok, f"{IM_START}assistant\n"),
        },
        "trunc_ids": _encode_no_special(tok, TRUNCATION_MARK),
        "newline_ids": _encode_no_special(tok, "\n"),
    }


def count_chat_tokens(tok: AutoTokenizer, consts: Dict[str, Any], messages: List[Dict[str, str]]) -> int:
    if not messages:
        return 0

    batch = [m.get("content", "") for m in messages]
    roles = [m.get("role", "") for m in messages]
    enc = tok(
        batch,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    total = 0
    for role, ids in zip(roles, enc["input_ids"]):
        total += len(consts["im_start_role_ids"].get(role, []))
        total += len(ids)
        total += len(consts["im_end_ids"])
    return total


def truncate_ids_to_text(tok: AutoTokenizer, ids: List[int], max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if len(ids) <= max_tokens:
        return tok.decode(ids, skip_special_tokens=True)
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


# =========================
# Docs preprocessing + prompt build
# =========================

def preprocess_documents(
    tok: AutoTokenizer,
    documents: List[Dict[str, Any]],
) -> Tuple[List[List[int]], List[List[int]], List[str], List[str]]:
    prefixes: List[str] = []
    all_texts: List[str] = []
    doc_id_list: List[str] = []
    original_texts: List[str] = []

    for doc in documents:
        doc_id = str(doc.get("id"))
        doc_id_list.append(doc_id)

        prefixes.append(f"[{doc_id}] ")

        txt = str(doc.get("text", "")).strip()
        # avoid collisions with "[n]" patterns inside doc text
        txt = re.sub(r"\[(\d+)\]", r"(\1)", txt).strip()
        original_texts.append(txt)
        all_texts.append(txt)

    prefix_encoded = tok(prefixes, add_special_tokens=False)["input_ids"]
    text_encoded = tok(all_texts, add_special_tokens=False)["input_ids"]

    return prefix_encoded, text_encoded, doc_id_list, original_texts


def build_user_prompt_with_truncation(
    tok: AutoTokenizer,
    consts: Dict[str, Any],
    user_template: str,
    system_prompt: str,
    query: str,
    documents: List[Dict[str, Any]],
    mode: str,
    preprocessed_docs: Optional[Tuple[List[List[int]], List[List[int]], List[str], List[str]]] = None,
) -> str:
    mode_tag = "/index" if mode == "index" else "/snapshot"
    suffix = "\n" + mode_tag

    placeholder_context = "<CTX>"
    skeleton = user_template.format(num=len(documents), question=query, context=placeholder_context)
    fixed_user_body = skeleton.replace(placeholder_context, "")

    fixed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fixed_user_body + suffix},
    ]
    fixed_tokens = count_chat_tokens(tok, consts, fixed_messages)
    remaining_for_context = PROMPT_TOKEN_BUDGET - fixed_tokens

    if remaining_for_context <= 0 or not documents:
        return user_template.format(num=len(documents), question=query, context="") + suffix

    if preprocessed_docs is None:
        preprocessed_docs = preprocess_documents(tok, documents)

    prefix_ids_list, doc_text_ids_list, doc_id_list, original_texts = preprocessed_docs
    nl_len = len(consts["newline_ids"])

    context_tokens = sum(len(p) + len(t) + nl_len for p, t in zip(prefix_ids_list, doc_text_ids_list))

    if context_tokens <= remaining_for_context:
        lines = [f"[{doc_id}] {txt}" for doc_id, txt in zip(doc_id_list, original_texts)]
        context_str = "\n".join(lines)
        return user_template.format(num=len(documents), question=query, context=context_str) + suffix

    # Need truncation
    n = len(documents)
    trunc_mark_tokens = len(consts["trunc_ids"])
    overhead_per_doc = [len(p) + trunc_mark_tokens + nl_len for p in prefix_ids_list]
    overhead_total = sum(overhead_per_doc)

    remaining_for_text_only = remaining_for_context - overhead_total
    if remaining_for_text_only <= 0:
        mark = TRUNCATION_MARK.strip()
        lines = [f"[{doc_id}] {mark}" for doc_id in doc_id_list]
        context_str = "\n".join(lines)
        return user_template.format(num=len(documents), question=query, context=context_str) + suffix

    avg_tokens_per_doc_text = max(1, remaining_for_text_only // n)

    lines: List[str] = []
    for doc_id, txt_ids in zip(doc_id_list, doc_text_ids_list):
        if len(txt_ids) <= avg_tokens_per_doc_text:
            txt = tok.decode(txt_ids, skip_special_tokens=True)
            lines.append(f"[{doc_id}] {txt}")
        else:
            keep_tokens = max(1, avg_tokens_per_doc_text - trunc_mark_tokens)
            shortened = truncate_ids_to_text(tok, txt_ids, keep_tokens).rstrip()
            lines.append(f"[{doc_id}] {shortened}{TRUNCATION_MARK}")

    context_str = "\n".join(lines)
    return user_template.format(num=len(documents), question=query, context=context_str) + suffix


# =========================
# Ranker output parsing
# =========================

def extract_visible_output_after_think(raw_output: str) -> str:
    if not raw_output:
        return ""
    marker = "</think>"
    idx = raw_output.rfind(marker)
    if idx == -1:
        return raw_output
    return raw_output[idx + len(marker):]


def parse_ranker_output(raw_output: str) -> List[str]:
    visible = extract_visible_output_after_think(raw_output)
    if not visible:
        return []
    indices = re.findall(r"\[(\d+)\]", visible)

    seen = set()
    ordered: List[str] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


# =========================
# IO helpers
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def load_descriptions(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Description file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_finished_ids(output_path: str) -> set:
    """
    Resume logic based on output field "id".
    """
    finished = set()
    if not os.path.exists(output_path):
        return finished
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            _id = rec.get("id")
            if _id is not None:
                finished.add(_id)
    return finished


# =========================
# API call
# =========================

def call_ranker_api(
    client: OpenAI,
    ranker_model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    resp = client.chat.completions.create(
        model=ranker_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def _build_doc_lookup(sample_docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping: doc_id(str) -> original doc dict
    """
    mp: Dict[str, Dict[str, Any]] = {}
    for d in (sample_docs or []):
        did = d.get("id")
        if did is None:
            continue
        mp[str(did)] = d
    return mp


def _selected_docs_from_ranked_ids(
    ranked_ids: List[str],
    doc_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert ranked doc ids -> list of {"id": ..., "text": ...} using original doc text.
    Skip ids not found in lookup.
    """
    out: List[Dict[str, Any]] = []
    for did in ranked_ids:
        d = doc_lookup.get(str(did))
        if not d:
            continue
        out.append({
            "id": d.get("id"),
            "text": d.get("text", ""),
        })
    return out


def process_one_sample(
    client: OpenAI,
    tok: AutoTokenizer,
    consts: Dict[str, Any],
    system_prompt: str,
    user_template: str,
    sample: Dict[str, Any],
    mode: str,
    ranker_model: str,
    api_try: int,
    top_k: int,
    max_output_tokens: int,
) -> Optional[Dict[str, Any]]:
    """
    Returns downstream-ready record:
      {
        "id": ...,
        "query": ...,
        "source": ...,
        "documents": [{"id":..., "text":...}, ...],
        "answer": ""
      }

    If API fails after retries: return None (skipped).
    If parsing fails: documents=[] but still returns a record.
    """
    _id = sample.get("id", sample.get("sample_id"))
    query = str(sample.get("query", "")).strip()
    std_answer = sample.get('answers', [''])[0].strip()
    source = sample.get("source")
    documents = sample.get("documents", []) or []

    # Prompt uses possibly truncated docs, but we map back to ORIGINAL docs in output
    pre_docs = preprocess_documents(tok, documents) if documents else None
    user_prompt = build_user_prompt_with_truncation(
        tok=tok,
        consts=consts,
        user_template=user_template,
        system_prompt=system_prompt,
        query=query,
        documents=documents,
        mode=mode,
        preprocessed_docs=pre_docs,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    doc_lookup = _build_doc_lookup(documents)

    last_err = None
    for _ in range(api_try):
        try:
            raw_output = call_ranker_api(
                client=client,
                ranker_model=ranker_model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.0,
            )
            ranked_ids = parse_ranker_output(raw_output)
            if top_k and top_k > 0:
                ranked_ids = ranked_ids[:top_k]

            selected_docs = _selected_docs_from_ranked_ids(ranked_ids, doc_lookup)

            # Downstream-ready output
            return {
                "id": _id,
                "query": query,
                "source": source,
                "documents": selected_docs,
                "std_answer": std_answer,
                "answer": "",
            }

        except Exception as e:
            last_err = e
            time.sleep(3)

    print(f"[API FAIL] id={_id} error={last_err} (skipped)")
    return None


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Run Ranker and output downstream-ready jsonl (selected docs mapped back to original texts)."
    )
    parser.add_argument("--input", required=True, help="Path to input test dataset jsonl")
    parser.add_argument("--lang", choices=["en", "zh"], required=True, help="Prompt language: en or zh")
    parser.add_argument("--mode", choices=["index", "snapshot"], required=True, help="Inference mode: index or snapshot")

    parser.add_argument("--tokenizer_path", required=True, help="Tokenizer path used for budgeting/truncation")
    parser.add_argument("--model_name", required=True, help="Ranker model name for chat.completions.create")
    parser.add_argument("--api_url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--output_path", default="selected_results/selected_docs.jsonl", help="Output jsonl path")

    parser.add_argument("--downstream_model", default="default", help="Downstream generator model name (for system prompt)")
    parser.add_argument("--desc_file_en", default=DEFAULT_DESC_FILE["en"], help="Path to English descriptions json")
    parser.add_argument("--desc_file_zh", default=DEFAULT_DESC_FILE["zh"], help="Path to Chinese descriptions json")

    parser.add_argument("--max_concurrency", type=int, default=1, help="Max worker threads")
    parser.add_argument("--api_try", type=int, default=5, help="API retry times on failure")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only top-k doc ids; 0 means keep all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for picking one description")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=False)
    consts = init_token_constants(tok)

    desc_file = args.desc_file_en if args.lang == "en" else args.desc_file_zh
    descs_by_model = load_descriptions(desc_file)
    if args.downstream_model not in descs_by_model:
        raise KeyError(f"Downstream model '{args.downstream_model}' not found in: {desc_file}")

    random.seed(args.seed)
    description = random.choice(descs_by_model[args.downstream_model])

    system_prompt = SYSTEM_TEMPLATES[args.lang].format(
        downstream_model=args.downstream_model,
        description=description,
    )
    user_template = USER_TEMPLATES[args.lang]

    samples = load_jsonl(args.input)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    finished = load_finished_ids(args.output_path)
    if finished:
        before = len(samples)
        samples = [s for s in samples if s.get("id", s.get("sample_id")) not in finished]
        print(f"[RESUME] skipping {before - len(samples)} finished; remaining {len(samples)}")

    if not samples:
        print("No remaining samples.")
        return

    client = OpenAI(api_key="EMPTY", base_url=args.api_url)

    processed = 0
    skipped_api_fail = 0
    FLUSH_INTERVAL = 1000

    with open(args.output_path, "a", encoding="utf-8", buffering=8 * 1024 * 1024) as fout:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as ex:
            futures = [
                ex.submit(
                    process_one_sample,
                    client,
                    tok,
                    consts,
                    system_prompt,
                    user_template,
                    sample,
                    args.mode,
                    args.model_name,
                    args.api_try,
                    args.top_k,
                    RANKER_MAX_OUTPUT_TOKENS,
                )
                for sample in samples
            ]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Running Ranker"):
                result = fut.result()
                if result is None:
                    skipped_api_fail += 1
                    continue

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed += 1

                if processed % FLUSH_INTERVAL == 0:
                    fout.flush()
                    os.fsync(fout.fileno())
                    print(f"[FLUSH] wrote {processed} records")

        fout.flush()
        os.fsync(fout.fileno())

    print(f"\nDone. Output: {args.output_path}")
    print(f"Written: {processed}")
    print(f"Skipped (API fail): {skipped_api_fail}")


if __name__ == "__main__":
    main()