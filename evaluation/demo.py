from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, AutoConfig
from openai import OpenAI
import openai


TRUNCATION_MARK = " <TRUNCATED>"

# Ranker budgeting
RANKER_MAX_CONTEXT_TOKENS = 40960
RANKER_MAX_OUTPUT_TOKENS = 4096
RANKER_PROMPT_TOKEN_BUDGET = RANKER_MAX_CONTEXT_TOKENS - RANKER_MAX_OUTPUT_TOKENS

IM_START = "<|im_start|>"
IM_END = "<|im_end|>\n"

# Downstream defaults
DOWNSTREAM_MAX_GEN_TOKENS = 8192
DOWNSTREAM_CALL_MAX_TOKENS = 2048

MODEL_CTX_OVERRIDES = {
    "gemma3": 131072,
}

# -------------------------
# Ranker prompt templates
# -------------------------
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


# =========================================================
# Ranker helpers (same behavior as your inference.py)
# =========================================================

def _encode_no_special(tok: AutoTokenizer, s: str) -> List[int]:
    return tok(s, add_special_tokens=False)["input_ids"]

def init_ranker_token_constants(tok: AutoTokenizer) -> Dict[str, Any]:
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

def count_ranker_chat_tokens(tok: AutoTokenizer, consts: Dict[str, Any], messages: List[Dict[str, str]]) -> int:
    if not messages:
        return 0
    batch = [m.get("content", "") for m in messages]
    roles = [m.get("role", "") for m in messages]
    enc = tok(batch, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)

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

def preprocess_documents(tok: AutoTokenizer, documents: List[Dict[str, Any]]) -> Tuple[List[List[int]], List[List[int]], List[str], List[str]]:
    prefixes: List[str] = []
    all_texts: List[str] = []
    doc_id_list: List[str] = []
    original_texts: List[str] = []

    for doc in documents:
        doc_id = str(doc.get("id"))
        doc_id_list.append(doc_id)
        prefixes.append(f"[{doc_id}] ")

        txt = str(doc.get("text", "")).strip()
        txt = re.sub(r"\[(\d+)\]", r"(\1)", txt).strip()
        original_texts.append(txt)
        all_texts.append(txt)

    prefix_encoded = tok(prefixes, add_special_tokens=False)["input_ids"]
    text_encoded = tok(all_texts, add_special_tokens=False)["input_ids"]
    return prefix_encoded, text_encoded, doc_id_list, original_texts

def build_ranker_user_prompt_with_truncation(
    tok: AutoTokenizer,
    consts: Dict[str, Any],
    user_template: str,
    system_prompt: str,
    query: str,
    documents: List[Dict[str, Any]],
    mode: str,
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
    fixed_tokens = count_ranker_chat_tokens(tok, consts, fixed_messages)
    remaining_for_context = RANKER_PROMPT_TOKEN_BUDGET - fixed_tokens

    if remaining_for_context <= 0 or not documents:
        return user_template.format(num=len(documents), question=query, context="") + suffix

    prefix_ids_list, doc_text_ids_list, doc_id_list, original_texts = preprocess_documents(tok, documents)
    nl_len = len(consts["newline_ids"])

    context_tokens = sum(len(p) + len(t) + nl_len for p, t in zip(prefix_ids_list, doc_text_ids_list))
    if context_tokens <= remaining_for_context:
        lines = [f"[{doc_id}] {txt}" for doc_id, txt in zip(doc_id_list, original_texts)]
        return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix

    # Need truncation
    n = len(documents)
    trunc_mark_tokens = len(consts["trunc_ids"])
    overhead_total = sum(len(p) + trunc_mark_tokens + nl_len for p in prefix_ids_list)

    remaining_for_text_only = remaining_for_context - overhead_total
    if remaining_for_text_only <= 0:
        mark = TRUNCATION_MARK.strip()
        lines = [f"[{doc_id}] {mark}" for doc_id in doc_id_list]
        return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix

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

    return user_template.format(num=len(documents), question=query, context="\n".join(lines)) + suffix

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

def _build_doc_lookup(sample_docs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    for d in (sample_docs or []):
        did = d.get("id")
        if did is None:
            continue
        mp[str(did)] = d
    return mp

def _selected_docs_from_ranked_ids(ranked_ids: List[str], doc_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for did in ranked_ids:
        d = doc_lookup.get(str(did))
        if not d:
            continue
        out.append({"id": d.get("id"), "text": d.get("text", "")})
    return out


# =========================================================
# Downstream helpers (same behavior as your downstream script)
# =========================================================

def init_downstream_tokenizer_and_budget(model_path: str, max_gen_tokens: int = DOWNSTREAM_MAX_GEN_TOKENS) -> Tuple[AutoTokenizer, int, int]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    mp = model_path.lower()

    override_val = None
    if model_type in MODEL_CTX_OVERRIDES:
        override_val = MODEL_CTX_OVERRIDES[model_type]
    else:
        for k, v in MODEL_CTX_OVERRIDES.items():
            if k.lower() in mp:
                override_val = v
                break

    candidates: List[int] = []
    for key in [
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "model_max_length",
        "max_seq_len",
        "max_sequence_length",
        "max_context_length",
        "context_length",
    ]:
        if hasattr(cfg, key):
            val = getattr(cfg, key)
            if isinstance(val, int) and val > 0:
                candidates.append(val)

    tok_mml = getattr(tok, "model_max_length", None)
    if isinstance(tok_mml, int) and 0 < tok_mml < 1_000_000:
        candidates.append(tok_mml)

    if override_val is not None:
        max_ctx = int(override_val)
    elif candidates:
        max_ctx = int(min(candidates))
    else:
        max_ctx = 32769

    budget = max_ctx - max_gen_tokens
    if budget <= 0:
        budget = max(256, max_ctx // 8)

    return tok, max_ctx, budget

def _down_count_chat_tokens(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> int:
    if hasattr(tok, "apply_chat_template"):
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)

    text = ""
    for m in messages:
        text += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
    return len(tok(text, add_special_tokens=True)["input_ids"])

def _down_count_text_tokens(tok: AutoTokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tok(text, add_special_tokens=False)["input_ids"])

def _down_truncate_text_to_tokens(tok: AutoTokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)

def build_downstream_prompt_with_truncation(
    tok: AutoTokenizer,
    prompt_budget: int,
    query: str,
    documents: List[Dict[str, Any]],
    lang: str = "en",
) -> str:
    doc_contents: List[str] = []
    for doc in documents:
        content = f"{doc.get('text', '')}".strip()
        doc_contents.append(content)

    placeholder = "<DOCS_PLACEHOLDER>"

    if lang == "zh":
        fixed_prompt = (
            "你将会收到一个问题和若干参考文档。请根据这些文档回答问题。\n\n"
            "指令：\n"
            "1）先分析问题并呈现清晰的推理过程。\n"
            "2）使用文档寻找相关证据。部分内容可能具有干扰性——请仔细阅读。\n"
            "3）严格遵循输出格式：先写“Reasoning”部分，然后给出以“Answer:”开头的最终答案。\n\n"
            "输入\n"
            f"- 问题: {query}\n"
            "- 文档:\n" + placeholder +
            "\n\n输出格式\n"
            "Reasoning: ...\n"
            "Answer: ..."
        )
    else:
        fixed_prompt = (
            "You are given a question and several reference documents. Please answer the question based on them.\n\n"
            "Instructions:\n"
            "1) Analyze the question first and present a clear line of reasoning.\n"
            "2) Use the documents to find relevant evidence. Some content may be distracting—read carefully.\n"
            "3) Follow the output format strictly: write the \"Reasoning\" section first, then provide the final answer prefixed with \"Answer:\".\n\n"
            "Input\n"
            f"- Question: {query}\n"
            "- Documents:\n" + placeholder +
            "\n\nOutput Format\n"
            "Reasoning: ...\n"
            "Answer: ..."
        )

    fixed_tokens = _down_count_chat_tokens(tok, [{"role": "user", "content": fixed_prompt.replace(placeholder, "")}])
    remaining_for_docs = prompt_budget - fixed_tokens

    if remaining_for_docs <= 0 or not documents:
        return fixed_prompt.replace(placeholder, "")

    def make_passages(contents: List[str]) -> str:
        if not contents:
            return "No relevant documents found."
        return "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contents)])

    # Try full
    full_prompt = fixed_prompt.replace(placeholder, make_passages(doc_contents))
    if _down_count_chat_tokens(tok, [{"role": "user", "content": full_prompt}]) <= prompt_budget:
        return full_prompt

    # Otherwise truncate evenly
    n = len(doc_contents)
    overhead_total = 0
    for i in range(n):
        prefix = f"[{i + 1}] "
        overhead_total += _down_count_text_tokens(tok, prefix + TRUNCATION_MARK + "\n")

    remaining_for_text_only = remaining_for_docs - overhead_total
    if remaining_for_text_only <= 0:
        truncated_contents = [TRUNCATION_MARK.strip()] * n
    else:
        avg_tokens = max(1, remaining_for_text_only // n)
        mark_tokens = _down_count_text_tokens(tok, TRUNCATION_MARK)
        truncated_contents = []
        for content in doc_contents:
            if _down_count_text_tokens(tok, content) <= avg_tokens:
                truncated_contents.append(content)
            else:
                keep = max(1, avg_tokens - mark_tokens)
                shortened = _down_truncate_text_to_tokens(tok, content, keep).rstrip()
                truncated_contents.append(shortened + TRUNCATION_MARK)

    # Final safety shrink
    prompt = fixed_prompt.replace(placeholder, make_passages(truncated_contents))
    while _down_count_chat_tokens(tok, [{"role": "user", "content": prompt}]) > prompt_budget:
        new_contents = []
        any_change = False
        for c in truncated_contents:
            t = _down_count_text_tokens(tok, c)
            if t > 1:
                new_c = _down_truncate_text_to_tokens(tok, c, t - 1)
                any_change = any_change or (new_c != c)
                new_contents.append(new_c)
            else:
                new_contents.append(c)
        if not any_change:
            break
        truncated_contents = new_contents
        prompt = fixed_prompt.replace(placeholder, make_passages(truncated_contents))

    return prompt

def parse_downstream_response(raw_response: str) -> Tuple[Optional[str], Optional[str]]:
    response_text = (raw_response or "").strip()
    if "**Answer:**" in response_text:
        parts = response_text.rsplit("**Answer:**", 1)
        reason = parts[0].replace("**Reasoning:**", "").replace("Reasoning:", "").strip()
        answer = parts[1].strip()
        return reason, answer
    if "Answer:" in response_text:
        parts = response_text.rsplit("Answer:", 1)
        reason = parts[0].replace("Reasoning:", "").strip()
        answer = parts[1].strip()
        return reason, answer
    return None, None

def _is_connection_like_error(e: Exception) -> bool:
    conn_types = (
        getattr(openai, "APIConnectionError", Exception),
        getattr(openai, "APITimeoutError", Exception),
        TimeoutError,
        ConnectionError,
    )
    if isinstance(e, conn_types):
        return True
    msg = (str(e) or "").lower()
    keywords = [
        "connection", "connect", "timeout", "timed out",
        "connection refused", "refused", "reset", "broken pipe",
        "name or service not known", "temporary failure in name resolution",
        "502", "503", "504", "bad gateway", "service unavailable", "gateway timeout"
    ]
    return any(k in msg for k in keywords)


# =========================================================
# Public demo class
# =========================================================

@dataclass
class Rank4GenDemoConfig:
    # Ranker
    ranker_api_base: str
    ranker_model: str
    ranker_tokenizer_path: str
    lang: str = "en"                 # en|zh
    mode: str = "index"              # index|snapshot
    downstream_model_for_ranker_desc: str = "default"
    desc_file_en: str = "model_descriptions/model_descriptions.json"
    desc_file_zh: str = "model_descriptions/model_descriptions_zh.json"
    seed: int = 42
    ranker_api_try: int = 3
    sleep_on_fail: float = 2.0

    # Downstream
    downstream_api_base: str = ""
    downstream_model: str = ""
    downstream_model_path: str = ""
    downstream_lang: Optional[str] = None
    downstream_api_try: int = 3


class Rank4GenDemo:
    """
    End-to-end demo wrapper:
      Ranker (set selection + ranking) -> Downstream generation

    Key point:
      - No top-k truncation. The downstream receives the full selected set (ordered).
    """

    def __init__(self, cfg: Rank4GenDemoConfig):
        self.cfg = cfg
        assert cfg.lang in ("en", "zh")
        assert cfg.mode in ("index", "snapshot")

        # Ranker tokenizer + constants
        self.ranker_tok = AutoTokenizer.from_pretrained(cfg.ranker_tokenizer_path, trust_remote_code=True, use_fast=False)
        self.ranker_consts = init_ranker_token_constants(self.ranker_tok)

        # Ranker system prompt
        desc_file = cfg.desc_file_en if cfg.lang == "en" else cfg.desc_file_zh
        descs_by_model = self._load_descriptions(desc_file)
        if cfg.downstream_model_for_ranker_desc not in descs_by_model:
            raise KeyError(f"downstream_model '{cfg.downstream_model_for_ranker_desc}' not in {desc_file}")

        random.seed(cfg.seed)
        description = random.choice(descs_by_model[cfg.downstream_model_for_ranker_desc])

        self.ranker_system_prompt = SYSTEM_TEMPLATES[cfg.lang].format(
            downstream_model=cfg.downstream_model_for_ranker_desc,
            description=description,
        )
        self.ranker_user_template = USER_TEMPLATES[cfg.lang]

        # Clients
        self.ranker_client = OpenAI(api_key="EMPTY", base_url=self._normalize_base_url(cfg.ranker_api_base))
        self.down_client = OpenAI(api_key="EMPTY", base_url=self._normalize_base_url(cfg.downstream_api_base)) if cfg.downstream_api_base else None

        # Downstream tokenizer/budget
        self.down_tok: Optional[AutoTokenizer] = None
        self.down_max_ctx: Optional[int] = None
        self.down_budget: Optional[int] = None
        if cfg.downstream_model_path:
            self.down_tok, self.down_max_ctx, self.down_budget = init_downstream_tokenizer_and_budget(cfg.downstream_model_path)

    @staticmethod
    def _normalize_base_url(api_base: str) -> str:
        s = (api_base or "").strip()
        if not s:
            raise ValueError("api_base is empty")
        if s.endswith("/v1"):
            return s
        return s.rstrip("/") + "/v1"

    @staticmethod
    def _load_descriptions(path: str) -> Dict[str, List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Description file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # Ranker step (NO top-k)
    # -------------------------
    def rank(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        docs = documents or []
        user_prompt = build_ranker_user_prompt_with_truncation(
            tok=self.ranker_tok,
            consts=self.ranker_consts,
            user_template=self.ranker_user_template,
            system_prompt=self.ranker_system_prompt,
            query=query,
            documents=docs,
            mode=self.cfg.mode,
        )

        messages = [
            {"role": "system", "content": self.ranker_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        doc_lookup = _build_doc_lookup(docs)

        last_err = None
        for _ in range(max(1, self.cfg.ranker_api_try)):
            try:
                resp = self.ranker_client.chat.completions.create(
                    model=self.cfg.ranker_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=RANKER_MAX_OUTPUT_TOKENS,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = resp.choices[0].message.content
                raw = raw if isinstance(raw, str) else str(raw)

                ranked_ids = parse_ranker_output(raw)  # <-- keep all selected ids (ordered)
                selected_docs = _selected_docs_from_ranked_ids(ranked_ids, doc_lookup)

                return {
                    "ranker_prompt_system": self.ranker_system_prompt,
                    "ranker_prompt_user": user_prompt,
                    "ranker_raw": raw,
                    "ranked_ids": ranked_ids,
                    "selected_docs": selected_docs,
                }
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.sleep_on_fail)

        raise RuntimeError(f"Ranker API failed after retries: {last_err}")

    # -------------------------
    # Downstream step
    # -------------------------
    def generate(self, query: str, selected_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.down_client is None:
            raise RuntimeError("downstream_api_base not configured")
        if not self.cfg.downstream_model:
            raise RuntimeError("downstream_model not configured")
        if self.down_tok is None or self.down_budget is None:
            raise RuntimeError("downstream_model_path not configured (needed for token budget)")

        lang = self.cfg.downstream_lang or self.cfg.lang
        prompt = build_downstream_prompt_with_truncation(
            tok=self.down_tok,
            prompt_budget=self.down_budget,
            query=query,
            documents=selected_docs or [],
            lang=lang,
        )

        last_err = None
        saw_conn = False
        saw_non_conn = False

        for attempt in range(max(1, self.cfg.downstream_api_try)):
            try:
                resp = self.down_client.chat.completions.create(
                    model=self.cfg.downstream_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=DOWNSTREAM_CALL_MAX_TOKENS,
                )
                raw = resp.choices[0].message.content
                raw = raw if isinstance(raw, str) else str(raw)

                reason, answer = parse_downstream_response(raw)
                if answer is not None:
                    return {
                        "downstream_prompt": prompt,
                        "downstream_raw": raw,
                        "reason": reason or "",
                        "answer": answer,
                        "status": "success",
                    }
                return {
                    "downstream_prompt": prompt,
                    "downstream_raw": raw,
                    "reason": "",
                    "answer": "",
                    "status": "failed_parsing",
                }

            except Exception as e:
                last_err = e
                is_conn = _is_connection_like_error(e)
                saw_conn = saw_conn or is_conn
                saw_non_conn = saw_non_conn or (not is_conn)
                time.sleep(min(60, 2 ** attempt))

        status = "failed_all_attempts" if (saw_non_conn and not saw_conn) else "failed_connection"
        return {
            "downstream_prompt": prompt,
            "downstream_raw": "",
            "reason": "",
            "answer": "",
            "status": status,
            "error": str(last_err),
        }

    # -------------------------
    # End-to-end
    # -------------------------
    def run(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        rank_out = self.rank(query=query, documents=documents)
        gen_out = self.generate(query=query, selected_docs=rank_out["selected_docs"])
        return {
            "query": query,
            "input_docs_count": len(documents or []),
            "ranked_ids": rank_out["ranked_ids"],
            "selected_docs": rank_out["selected_docs"],   # <-- full selected set
            "answer": gen_out.get("answer", ""),
            "status": gen_out.get("status", "unknown"),
            # demo/debug fields
            "ranker_prompt_user": rank_out["ranker_prompt_user"],
            "ranker_raw": rank_out["ranker_raw"],
            "downstream_prompt": gen_out.get("downstream_prompt", ""),
            "downstream_raw": gen_out.get("downstream_raw", ""),
        }

    def run_jsonl(self, input_path: str, output_path: str) -> None:
        """
        Input jsonl line:
          {"id":..., "query":..., "documents":[{"id":..,"text":..}, ...], "answers":[...], "source":...}

        Output jsonl line (evaluation-ready):
          {"id":..., "query":..., "source":..., "documents":[selected...],
           "std_answer":..., "answer":..., "status":...}
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                _id = rec.get("id", rec.get("sample_id"))
                query = str(rec.get("query", "")).strip()
                documents = rec.get("documents", []) or []

                std_answer = ""
                answers = rec.get("answers", [""])
                if isinstance(answers, list) and answers:
                    std_answer = str(answers[0] or "").strip()
                elif isinstance(answers, str):
                    std_answer = answers.strip()

                try:
                    out = self.run(query=query, documents=documents)
                    fout.write(json.dumps({
                        "id": _id,
                        "query": query,
                        "source": rec.get("source"),
                        "documents": out["selected_docs"],
                        "std_answer": std_answer,
                        "answer": out.get("answer", ""),
                        "status": out.get("status", "unknown"),
                    }, ensure_ascii=False) + "\n")
                except Exception as e:
                    fout.write(json.dumps({
                        "id": _id,
                        "query": query,
                        "source": rec.get("source"),
                        "documents": [],
                        "std_answer": std_answer,
                        "answer": "",
                        "status": "failed_pipeline",
                        "error": str(e),
                    }, ensure_ascii=False) + "\n")
