#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downstream RAG Answer Evaluation Script

This script evaluates a downstream generator model in a RAG setting:
- Given a query and candidate documents (with rerank scores),
- It builds a single-turn user prompt (English/Chinese),
- Automatically computes prompt token budget from HF config/tokenizer,
- Truncates documents to fit the budget,
- Calls an OpenAI-compatible chat completion endpoint,
- Parses "Reasoning" and "Answer" sections,
- Writes jsonl results with resume + output cleaning.

Key changes from the original:
- Accepts only ONE API endpoint: --api_endpoint
- Strategy is fixed to "all" (no CLI arg)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from openai import OpenAI
import openai


# =========================
# Global configuration
# =========================

MAX_GEN_TOKENS = 8192
TRUNCATION_MARK = " <TRUNCATED>"

TOKENIZER = None
MODEL_MAX_CTX = None
PROMPT_BUDGET = None

# Optional overrides for models that do not expose a reliable max context length in config.
MODEL_CTX_OVERRIDES = {
    # key can be cfg.model_type OR a substring of --model_path
    "gemma3": 131072,  # 128k
}

# Output statuses we keep in the output file.
KEEP_STATUSES = {"success", "failed_parsing", "failed_all_attempts"}


# =========================
# Token budget estimation
# =========================

def init_tokenizer_and_budget(model_path: str, max_gen_tokens: int = MAX_GEN_TOKENS) -> None:
    """
    Initialize tokenizer/config and infer model max context length, then compute prompt budget:
      prompt_budget = max_ctx - max_gen_tokens

    Priority:
      1) MODEL_CTX_OVERRIDES
      2) config/tokenizer candidates
      3) rope_scaling heuristic fallback
      4) default fallback (32k-ish)
    """
    global TOKENIZER, MODEL_MAX_CTX, PROMPT_BUDGET

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    source = None
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    mp = model_path.lower()

    # 0) Overrides
    override_val = None
    if model_type in MODEL_CTX_OVERRIDES:
        override_val = MODEL_CTX_OVERRIDES[model_type]
        source = f"override(model_type={model_type})"
    else:
        for k, v in MODEL_CTX_OVERRIDES.items():
            if k.lower() in mp:
                override_val = v
                source = f"override(model_path contains '{k}')"
                break

    # 1) config/tokenizer candidates
    candidates: List[int] = []
    cand_sources: List[str] = []

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
                cand_sources.append(f"config.{key}={val}")

    tok_mml = getattr(TOKENIZER, "model_max_length", None)
    if isinstance(tok_mml, int) and 0 < tok_mml < 1_000_000:
        candidates.append(tok_mml)
        cand_sources.append(f"tokenizer.model_max_length={tok_mml}")

    # 2) rope_scaling fallback (best-effort)
    if not candidates:
        text_cfg = getattr(cfg, "text_config", None)
        rope_scaling = None

        if isinstance(text_cfg, dict):
            rope_scaling = text_cfg.get("rope_scaling")
        else:
            rope_scaling = getattr(text_cfg, "rope_scaling", None)

        factor = None
        if isinstance(rope_scaling, dict):
            factor = rope_scaling.get("factor", None)

        if isinstance(factor, (int, float)) and factor > 0:
            base_ctx = 8192
            rope_guess = int(base_ctx * float(factor))
            candidates.append(rope_guess)
            cand_sources.append(f"rope_scaling(base={base_ctx} * factor={factor})={rope_guess}")

    # 3) choose max_ctx
    if override_val is not None:
        MODEL_MAX_CTX = int(override_val)
    elif candidates:
        MODEL_MAX_CTX = int(min(candidates))
        source = source or f"auto(min of {', '.join(cand_sources)})"
    else:
        MODEL_MAX_CTX = 32769
        source = "default(32k)"

    # 4) compute budget
    PROMPT_BUDGET = MODEL_MAX_CTX - max_gen_tokens
    if PROMPT_BUDGET <= 0:
        PROMPT_BUDGET = max(256, MODEL_MAX_CTX // 8)
        print(f"[Warn] Bad budget; fallback: max_ctx={MODEL_MAX_CTX}, max_gen={max_gen_tokens} -> budget={PROMPT_BUDGET}")

    print(f"[Tokenizer] Loaded from: {model_path}")
    print(f"[MaxCtx] model_type={model_type}, max_ctx={MODEL_MAX_CTX}, source={source}")
    if candidates:
        print(f"[MaxCtx] candidates={candidates} ({'; '.join(cand_sources)})")
    print(f"[Budget] max_gen_tokens={max_gen_tokens}, prompt_budget={PROMPT_BUDGET}")


def count_chat_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Count tokens for chat messages using tokenizer.apply_chat_template if available,
    otherwise fall back to a simple role/content concatenation.
    """
    if TOKENIZER is None:
        raise RuntimeError("TOKENIZER not initialized. Call init_tokenizer_and_budget() first.")

    if hasattr(TOKENIZER, "apply_chat_template"):
        ids = TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return len(ids)

    text = ""
    for m in messages:
        text += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
    return len(TOKENIZER(text, add_special_tokens=True)["input_ids"])


def count_text_tokens(text: str) -> int:
    """Count tokens for plain text (no chat template wrappers)."""
    if not text:
        return 0
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate plain text to max_tokens using tokenizer token ids."""
    if max_tokens <= 0 or not text:
        return ""
    ids = TOKENIZER(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return TOKENIZER.decode(ids[:max_tokens], skip_special_tokens=True)


# =========================
# Client & parsing helpers
# =========================

def setup_client(api_endpoint: str) -> OpenAI:
    """
    Create a single OpenAI client. api_endpoint can be:
      - "IP:PORT"
      - "http://IP:PORT"
      - "http://IP:PORT/v1"
    """
    ep = api_endpoint.strip()
    if ep.startswith("http://") or ep.startswith("https://"):
        # If user provides full base_url, accept it.
        if ep.endswith("/v1"):
            base_url = ep
        else:
            base_url = ep.rstrip("/") + "/v1"
    else:
        # Treat as IP:PORT
        base_url = f"http://{ep}/v1"

    print(f"[Client] base_url={base_url}")
    return OpenAI(api_key="EMPTY", base_url=base_url)


def parse_response(raw_response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse response into (reasoning, answer).
    Supports:
      - "Reasoning: ... Answer: ..."
      - "**Reasoning:** ... **Answer:** ..."
    """
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


# =========================
# Prompt building (fixed strategy = all)
# =========================

def build_prompt_with_truncation(query: str, documents: List[Dict[str, Any]], lang: str = "en") -> str:
    """
    Build a prompt with the same format as before, but truncate doc contents if the
    total prompt exceeds PROMPT_BUDGET.
    """
    # Build doc contents
    doc_contents: List[str] = []
    for doc in documents:
        content = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
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
            "- 文档:\n" +
            placeholder +
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
            "- Documents:\n" +
            placeholder +
            "\n\nOutput Format\n"
            "Reasoning: ...\n"
            "Answer: ..."
        )

    fixed_messages = [{"role": "user", "content": fixed_prompt.replace(placeholder, "")}]
    fixed_tokens = count_chat_tokens(fixed_messages)

    remaining_for_docs = PROMPT_BUDGET - fixed_tokens
    if remaining_for_docs <= 0 or not documents:
        # If no budget, return minimal prompt (still includes docs untruncated)
        return fixed_prompt.replace(placeholder, "")

    def make_passages(contents: List[str]) -> str:
        if not contents:
            return "No relevant documents found."
        lines = []
        for i, c in enumerate(contents):
            lines.append(f"[{i + 1}] {c}")
        return "\n".join(lines)

    # Try full prompt first
    full_passages = make_passages(doc_contents)
    full_prompt = fixed_prompt.replace(placeholder, full_passages)
    if count_chat_tokens([{"role": "user", "content": full_prompt}]) <= PROMPT_BUDGET:
        return full_prompt

    # Otherwise, truncate per doc equally (token-based)
    n = len(doc_contents)
    per_doc_overhead = []
    for i in range(n):
        prefix = f"[{i + 1}] "
        per_doc_overhead.append(count_text_tokens(prefix + TRUNCATION_MARK + "\n"))

    overhead_total = sum(per_doc_overhead)
    remaining_for_text_only = remaining_for_docs - overhead_total

    if remaining_for_text_only <= 0:
        truncated_contents = [TRUNCATION_MARK.strip()] * n
    else:
        avg_tokens = max(1, remaining_for_text_only // n)
        mark_tokens = count_text_tokens(TRUNCATION_MARK)

        truncated_contents = []
        for content in doc_contents:
            if count_text_tokens(content) <= avg_tokens:
                truncated_contents.append(content)
            else:
                keep = max(1, avg_tokens - mark_tokens)
                shortened = truncate_text_to_tokens(content, keep).rstrip()
                truncated_contents.append(shortened + TRUNCATION_MARK)

    # Final safety loop: shrink until within budget
    passages = make_passages(truncated_contents)
    prompt = fixed_prompt.replace(placeholder, passages)

    while count_chat_tokens([{"role": "user", "content": prompt}]) > PROMPT_BUDGET:
        new_contents = []
        any_change = False
        for c in truncated_contents:
            t = count_text_tokens(c)
            if t > 1:
                new_c = truncate_text_to_tokens(c, t - 1)
                any_change = any_change or (new_c != c)
                new_contents.append(new_c)
            else:
                new_contents.append(c)
        if not any_change:
            break
        truncated_contents = new_contents
        passages = make_passages(truncated_contents)
        prompt = fixed_prompt.replace(placeholder, passages)

    return prompt


# =========================
# API calling with retries + temperature schedule
# =========================

def _parse_temperature_list(s: str) -> List[float]:
    """
    Accepts:
      - "0,0.3,0.5,0.7,1.0"
      - "0 0.3 0.5 0.7 1.0"
    Returns a sorted (ascending) list.
    """
    if s is None:
        return [0.0]
    s = s.strip()
    if not s:
        return [0.0]

    parts = [p.strip() for p in (s.split(",") if "," in s else s.split()) if p.strip()]
    temps = [float(x) for x in parts] if parts else [0.0]
    return sorted(temps) if temps else [0.0]


def call_api(client: OpenAI, model_name: str, prompt: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=float(temperature),
        max_tokens=2048,
    )
    content = resp.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def _is_connection_like_error(e: Exception) -> bool:
    """
    Best-effort classification of connection-like errors.
    """
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


def process_entry(
    client: OpenAI,
    data: Dict[str, Any],
    model_name: str,
    api_try: int,
    lang: str,
    temperature_list: List[float],
    _max_try: Optional[int] = None,
    _saw_conn_error: bool = False,
    _saw_non_conn_error: bool = False,
) -> Dict[str, Any]:
    """
    Retry logic:
      - attempt index increases 0,1,2...
      - temperature follows temperature_list (clipped to last element)
      - exponential backoff: 1,2,4,... up to 60s

    On retries exhausted:
      - if we saw only non-connection errors => failed_all_attempts (kept)
      - otherwise => failed_connection (NOT kept, so next run can retry)
    """
    if _max_try is None:
        _max_try = api_try

    result = data.copy()
    # Clean old fields if present
    for k in ["status", "error", "raw-response", "reason", "answer", "final_prompt"]:
        result.pop(k, None)

    # Strategy fixed to ALL docs
    documents = data.get("documents", []) or []
    query = data.get("query", "") or ""
    prompt = build_prompt_with_truncation(query, documents, lang=lang)
    result["final_prompt"] = prompt

    if api_try <= 0:
        result["reason"] = ""
        result["answer"] = ""

        if _saw_non_conn_error and not _saw_conn_error:
            result["error"] = "ALL_ATTEMPTS_FAILED_NON_CONNECTION"
            result["status"] = "failed_all_attempts"
        elif _saw_non_conn_error and _saw_conn_error:
            result["error"] = "API_RETRIES_EXHAUSTED_MIXED_ERRORS"
            result["status"] = "failed_connection"
        else:
            result["error"] = "API_RETRIES_EXHAUSTED_CONNECTION"
            result["status"] = "failed_connection"

        return result

    attempt_index = (_max_try - api_try)
    if not temperature_list:
        temperature_list = [0.0]
    temp_idx = min(attempt_index, len(temperature_list) - 1)
    temperature = float(temperature_list[temp_idx])

    try:
        raw = call_api(client, model_name, prompt, temperature=temperature)
        result["raw-response"] = raw

        if raw is None:
            return process_entry(
                client, data, model_name, api_try - 1, lang, temperature_list,
                _max_try=_max_try,
                _saw_conn_error=_saw_conn_error,
                _saw_non_conn_error=True,
            )

        reason, answer = parse_response(raw)
        if answer is not None:
            result["reason"] = reason
            result["answer"] = answer
            result["status"] = "success"
        else:
            result["reason"] = ""
            result["answer"] = ""
            result["status"] = "failed_parsing"

        return result

    except openai.BadRequestError as e:
        # Non-connection error
        attempt = (_max_try - api_try)
        backoff = min(60, 2 ** attempt)
        print(f"[Warn] ID {data.get('id', 'N/A')} BadRequestError: {e} (temp={temperature}) backoff={backoff}s")
        time.sleep(backoff)
        return process_entry(
            client, data, model_name, api_try - 1, lang, temperature_list,
            _max_try=_max_try,
            _saw_conn_error=_saw_conn_error,
            _saw_non_conn_error=True,
        )

    except Exception as e:
        is_conn = _is_connection_like_error(e)
        attempt = (_max_try - api_try)
        backoff = min(60, 2 ** attempt)
        print(
            f"[Error] ID {data.get('id', 'N/A')} API fail: {e}. "
            f"Retry in {backoff}s (attempt={attempt+1}/{_max_try}, temp={temperature}, conn_like={is_conn})"
        )
        time.sleep(backoff)
        return process_entry(
            client, data, model_name, api_try - 1, lang, temperature_list,
            _max_try=_max_try,
            _saw_conn_error=(_saw_conn_error or is_conn),
            _saw_non_conn_error=(_saw_non_conn_error or (not is_conn)),
        )


# =========================
# Output cleaning + resume
# =========================

def clean_existing_output(output_path: str) -> Tuple[Dict[str, str], set]:
    """
    Keep only KEEP_STATUSES lines; for duplicate id, keep the last.
    Rewrite the output file to remove dirty lines.
    """
    kept_by_id: Dict[str, str] = {}
    if not os.path.exists(output_path):
        return kept_by_id, set()

    print(f"[Clean] scanning existing output: {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                _id = d.get("id")
                if not _id:
                    continue
                if d.get("status") in KEEP_STATUSES:
                    kept_by_id[_id] = line
            except Exception:
                continue

    processed_ids = set(kept_by_id.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        for line in kept_by_id.values():
            if not line.endswith("\n"):
                line += "\n"
            f.write(line)

    print(f"[Clean] kept {len(processed_ids)} records (success/failed_parsing/failed_all_attempts).")
    return kept_by_id, processed_ids


# =========================
# Main loop
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream RAG answer evaluation.")
    parser.add_argument("--data_path", type=str, default="selected_results/selected_docs.jsonl", help="Input dataset jsonl path")
    parser.add_argument("--output_dir", type=str, default="downstream_results", help="Output directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (for output filename)")
    parser.add_argument("--model_name", type=str, required=True, help="Downstream model name (for API request)")
    parser.add_argument("--api_endpoint", type=str, required=True, help="Single API endpoint (IP:PORT or http://..)")
    parser.add_argument("--model_path", type=str, required=True, help="HF model path for tokenizer/config budget inference")

    parser.add_argument("--max_concurrency", type=int, default=1, help="Max concurrency threads")
    parser.add_argument("--api_try", type=int, default=5, help="Max retry attempts per sample")
    parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Prompt language")
    parser.add_argument(
        "--temperature_list",
        type=str,
        default="0,0.3,0.7,1.0",
        help="Temperature schedule, comma or space separated (ascending). e.g. 0,0.3,0.7,1.0",
    )

    args = parser.parse_args()
    args.temperature_list = _parse_temperature_list(args.temperature_list)
    print(f"[TempList] {args.temperature_list}")

    init_tokenizer_and_budget(args.model_path, max_gen_tokens=MAX_GEN_TOKENS)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset_name}-{args.model_name}.jsonl")

    # Clean existing output and resume
    _, processed_ids = clean_existing_output(output_path)

    # Load tasks
    all_tasks: List[Dict[str, Any]] = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_tasks.append(json.loads(line))
            except Exception:
                continue
    print(f"[Load] loaded {len(all_tasks)} tasks from {args.data_path}")

    tasks_to_run = [t for t in all_tasks if t.get("id") not in processed_ids]
    if not tasks_to_run:
        print("[Done] no remaining tasks.")
        return

    print(f"[Run] remaining tasks: {len(tasks_to_run)}")

    # Single client
    client = setup_client(args.api_endpoint)

    BATCH_SIZE = 100
    FLUSH_INTERVAL = 1

    with open(output_path, "a", encoding="utf-8") as f_out:
        pbar = tqdm(total=len(tasks_to_run), desc="Progress [all]")

        for batch_start in range(0, len(tasks_to_run), BATCH_SIZE):
            batch = tasks_to_run[batch_start: batch_start + BATCH_SIZE]

            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        process_entry,
                        client,
                        data,
                        args.model_name,
                        args.api_try,
                        args.lang,
                        args.temperature_list,
                    ): data
                    for data in batch
                }

                batch_count = 0
                for fut in as_completed(futures):
                    res = fut.result()
                    status = res.get("status", "unknown")

                    if status in KEEP_STATUSES:
                        f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    else:
                        # failed_connection is intentionally not written (so next run can retry)
                        print(f"[Info] ID {res.get('id')} status={status} not written (retry next run).")

                    batch_count += 1
                    pbar.update(1)

                    if batch_count % FLUSH_INTERVAL == 0:
                        f_out.flush()

            f_out.flush()

        pbar.close()

    print(f"[Done] output saved to: {output_path}")


if __name__ == "__main__":
    main()
