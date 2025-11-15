import argparse
import os
import re
import sys
import time
from typing import List, Optional, Tuple, Any

# Output behavior flags (can be overridden by CLI)
OVERWRITE_FLAG = False
RESUME_FLAG = False

# === User Config (edit here) ===
CONFIG = {
    "BASE_URL": "http://localhost:1235/v1",   # 接口地址：LM Studio/OpenAI 兼容服务的 Base URL（端口改这里，例如 1234/1235）
    "API_KEY": "lm-studio",                    # API 密钥：LM Studio 通常接受任意非空字符串
    "MODEL": "huihui-hunyuan-mt-chimera-7b-abliterated-i1",  # 模型名称：在 LM Studio 中暴露的模型名
    "INPUT": "whole.txt",                      # 输入文件：待翻译的原文文件路径
    "OUTPUT": "translated.txt",                # 输出文件：翻译结果写入的文件路径
    "SOURCE_LANG": "auto",                     # 源语言：auto 自动检测，或指定如 'en'、'ja' 等
    "TARGET_LANG": "zh",                       # 目标语言：如 'zh'（中文）、'en'（英文）
    "CHUNK_SIZE_CHARS": 3000,                   # 分块最大字符数：控制每段请求的长度（过大易超时，过小上下文不足）
    "OVERLAP_CHARS": 120,                       # 分块重叠字符数：用于上下文衔接（避免句子被切断导致误译）
    "OVERWRITE": False,                          # 是否覆盖输出文件：True 覆盖，False 不覆盖
    "RESUME": True,                              # 是否追加续写：True 追加（常用于断点续跑），False 不追加
    "CONTEXT_ONLY_OVERLAP": True,                # 仅把重叠部分作为上下文（不翻译不输出），提高连贯性
    "START_CHUNK": 1,                            # 起始分块序号（从 1 开始）：用于跳过已完成部分继续翻译
    "REQUEST_TIMEOUT": 90,                       # 单次请求超时时间（秒）：过短易超时，过长遇到异常可能等待久
    "MAX_RETRIES": 3,                            # 每段失败重试次数：请求异常/超时的重试次数
    "SKIP_ON_ERROR": True,                       # 失败后是否自动跳过：True 写入失败标记继续，False 询问是否跳过
    "USE_STREAM": True,                          # 是否使用流式输出：一般更稳妥，减少长响应卡住的概率
    # 失败自动二分设置
    "AUTO_BISECT_ON_FAIL": True,                 # 启用失败自动二分：将失败段按语义点一分为二重试，递归直至上限
    "BISECT_MIN_CHARS": 600,                     # 二分最小长度：内容长度小于等于该值时不再继续二分
    "BISECT_MAX_DEPTH": 3,                       # 二分最大深度：限制递归层级，防止无限拆分
    # 输出后处理：隐藏思维链与背景标记清理
    "HIDE_CHAIN": False,                          # 是否隐藏思维链（如 <think>...</think>）
    "CHAIN_TAG": "think",                        # 思维链包裹标签名（可自定义，如 "thinking" 映射为 <thinking>..</thinking>）
    # PDF 识别默认配置
    "PDF_RECOGNIZER": "auto",
    "VLM_URL": "",
    "VLM_KEY": "",
    "PDF_DPI": 200,
    "PDF_PAGES": "",
}
# === End User Config ===


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter supporting Chinese and Western punctuation and preserving code blocks."""
    text = text.replace("\r\n", "\n")
    code_blocks = re.split(r"(```[\s\S]*?```)", text)
    sentences: List[str] = []
    sentence_end_pattern = re.compile(r"(?<=[。！？!?.])\s+|\n+")
    for block in code_blocks:
        if block.startswith("```") and block.endswith("```"):
            sentences.append(block)
        else:
            parts = [p for p in re.split(sentence_end_pattern, block) if p and not p.isspace()]
            sentences.extend(parts)
    return sentences


def chunk_by_chars(sentences: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """Group sentences into chunks constrained by character count with optional overlap (content includes overlap)."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        s_len = len(sent)
        if current and current_len + s_len > max_chars:
            chunk_text = "".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            overlap = chunk_text[-overlap_chars:] if overlap_chars > 0 else ""
            current = [overlap] if overlap else []
            current_len = len(overlap)
        current.append(sent)
        current_len += s_len
    if current:
        chunk_text = "".join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def chunk_by_chars_pairs(sentences: List[str], max_chars: int, overlap_chars: int) -> List[tuple[str, str]]:
    """Group sentences into (context, content) pairs.
    - context: overlap-only prefix used for understanding; MUST NOT be translated/output
    - content: the real content to translate and write out
    The length budget (max_chars) is applied to len(context + content), matching the non-pair logic.
    """
    pairs: List[tuple[str, str]] = []
    current_context = ""
    current_content: List[str] = []
    current_len = len(current_context)

    for sent in sentences:
        s_len = len(sent)
        if current_content and current_len + s_len > max_chars:
            # finalize current pair
            content_text = "".join(current_content).strip()
            if content_text:
                pairs.append((current_context, content_text))
            # compute next context from combined text
            combined = (current_context + content_text)
            next_context = combined[-overlap_chars:] if overlap_chars > 0 else ""
            current_context = next_context
            current_content = []
            current_len = len(current_context)
        # add sentence to content (even if it exceeds, we accept single long sentence)
        current_content.append(sent)
        current_len += s_len

    if current_content:
        content_text = "".join(current_content).strip()
        if content_text:
            pairs.append((current_context, content_text))
    return pairs


def build_messages(source_lang: str, target_lang: str, chunk: str, system_prompt: Optional[str] = None) -> list:
    system_prompt_default = (
        f"You are a professional translator. Translate the user's text from {source_lang or 'the original language'} "
        f"to {target_lang}. Preserve original formatting, markdown structure, and line breaks. "
        f"Do not add explanations or comments. If there are code blocks or inline code, keep them unchanged."
    )
    sys_prompt = system_prompt if (system_prompt and system_prompt.strip()) else system_prompt_default
    user_prompt = "Translate the following content. Output only the translation.\n\n" + chunk
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_with_context(source_lang: str, target_lang: str, context: str, content: str, system_prompt: Optional[str] = None) -> list:
    """Build chat messages where 'context' is for understanding only and must not be translated/output."""
    system_prompt_default = (
        f"You are a professional translator. Translate only the text in the 'Content' section from {source_lang or 'the original language'} "
        f"to {target_lang}. Use the 'Context' section only for understanding. Do NOT translate or include the Context in the output. "
        f"Preserve formatting, markdown structure, and line breaks in the translation."
    )
    sys_prompt = system_prompt if (system_prompt and system_prompt.strip()) else system_prompt_default
    user_prompt = (
        "Here is additional context and the content to translate.\n\n"
        "Context (for reference only, DO NOT translate or output):\n" +
        (f"```\n{context}\n```\n\n" if context else "(none)\n\n") +
        "Content to translate (OUTPUT ONLY the translation of this section):\n" +
        f"```\n{content}\n```"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


def translate_chunk(
    client: Any,
    model: str,
    messages: list,
    retries: int = 3,
    retry_delay: float = 2.0,
    timeout: int | float | None = None,
    use_stream: bool = True,
    skip_on_error: bool = False,
    chunk_index: int | None = None,
    *,
    temperature: float = 0.0,
    top_p: Optional[float] = 1.0,
    top_k: Optional[int] = 0,
    repetition_penalty: Optional[float] = 0.0,
    length_penalty: Optional[float] = 0.0,
) -> str:
    last_exc: Exception | None = None
    # Build vendor-specific extras
    def _extra_body():
        extra: dict = {}
        if top_k and top_k > 0:
            extra["top_k"] = int(top_k)
        if repetition_penalty and repetition_penalty > 0:
            extra["repetition_penalty"] = float(repetition_penalty)
        if length_penalty and length_penalty != 0:
            extra["length_penalty"] = float(length_penalty)
        return extra
    for attempt in range(1, retries + 1):
        try:
            if use_stream:
                acc = []
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                    timeout=timeout,
                    extra_body=_extra_body(),
                )
                for event in stream:  # ChatCompletionChunk events
                    try:
                        delta = event.choices[0].delta
                        piece = getattr(delta, "content", None)
                        if piece:
                            acc.append(piece)
                    except Exception:
                        # best-effort accumulate
                        pass
                return "".join(acc)
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    extra_body=_extra_body(),
                )
                content = resp.choices[0].message.content or ""
                return content
        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"Warning: chunk {chunk_index or ''} attempt {attempt} failed: {e}. Retrying...")
                time.sleep(retry_delay * attempt)
            else:
                if skip_on_error:
                    marker = f"<<<FAILED_CHUNK_{chunk_index or '?'}>>>"
                    print(f"Error: {marker} - skipping after {retries} attempts.")
                    return marker
    raise RuntimeError(f"Translation request failed after {retries} attempts: {last_exc}")


# ===== Helpers for auto-bisect and output replacement =====
FAILED_MARKER_FMT = "<<<FAILED_CHUNK_{idx}>>>"


def find_split_point(s: str) -> int:
    """Find a split point near the middle at punctuation to keep semantics. Fallback to exact middle."""
    n = len(s)
    if n < 2:
        return n // 2
    mid = n // 2
    # search window
    window = 200
    punct = set("。！？!?.\n")
    # prefer right side first to keep earlier half shorter
    for offset in range(0, window + 1):
        r = mid + offset
        if r < n and s[r] in punct:
            return r + 1
        l = mid - offset
        if l > 0 and s[l] in punct:
            return l + 1
    return mid


def try_translate_unit(
    client: Any,
    model: str,
    source_lang: str,
    target_lang: str,
    context: str,
    content: str,
    request_timeout: int,
    max_retries: int,
    use_stream: bool,
    chunk_index: int,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = 1.0,
    top_k: Optional[int] = 0,
    repetition_penalty: Optional[float] = 0.0,
    length_penalty: Optional[float] = 0.0,
) -> str:
    messages = build_messages_with_context(source_lang, target_lang, context, content, system_prompt=system_prompt)
    return translate_chunk(
        client, model, messages,
        retries=max_retries, timeout=request_timeout, use_stream=use_stream,
        skip_on_error=False, chunk_index=chunk_index,
        temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, length_penalty=length_penalty,
    )


def translate_with_bisect(
    client: Any,
    model: str,
    source_lang: str,
    target_lang: str,
    context: str,
    content: str,
    request_timeout: int,
    max_retries: int,
    use_stream: bool,
    chunk_index: int,
    min_chars: int,
    max_depth: int,
    depth: int = 0,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = 1.0,
    top_k: Optional[int] = 0,
    repetition_penalty: Optional[float] = 0.0,
    length_penalty: Optional[float] = 0.0,
) -> Optional[str]:
    if depth >= max_depth or len(content) <= min_chars:
        return None
    split_at = find_split_point(content)
    left = content[:split_at]
    right = content[split_at:]
    left_out: Optional[str] = None
    right_out: Optional[str] = None
    try:
        left_out = try_translate_unit(
            client, model, source_lang, target_lang, context, left,
            request_timeout, max_retries, use_stream, chunk_index,
            system_prompt=system_prompt,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
        )
    except Exception:
        # bisect left further
        left_out = translate_with_bisect(
            client, model, source_lang, target_lang, context, left,
            request_timeout, max_retries, use_stream, chunk_index,
            min_chars, max_depth, depth + 1,
            system_prompt=system_prompt,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
        )
    try:
        right_out = try_translate_unit(
            client, model, source_lang, target_lang, context, right,
            request_timeout, max_retries, use_stream, chunk_index,
            system_prompt=system_prompt,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
        )
    except Exception:
        right_out = translate_with_bisect(
            client, model, source_lang, target_lang, context, right,
            request_timeout, max_retries, use_stream, chunk_index,
            min_chars, max_depth, depth + 1,
            system_prompt=system_prompt,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
        )
    if left_out is not None and right_out is not None:
        # Join with a newline boundary to avoid accidental concatenation
        joiner = "\n" if (left_out and right_out and not left_out.endswith("\n")) else ""
        return f"{left_out}{joiner}{right_out}"
    return None


def output_contains_failed_marker(output_path: str, idx: int) -> bool:
    if not os.path.exists(output_path):
        return False
    marker = FAILED_MARKER_FMT.format(idx=idx)
    legacy_pattern = re.compile(rf"【第{idx}段翻译失败：[^】]*】")
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = f.read()
        if marker in data:
            return True
        if legacy_pattern.search(data) is not None:
            return True
        return False
    except Exception:
        return False


def replace_failed_marker_in_output(output_path: str, idx: int, text: str) -> bool:
    marker = FAILED_MARKER_FMT.format(idx=idx)
    legacy_pattern = re.compile(rf"【第{idx}段翻译失败：[^】]*】")
    combined_pattern = re.compile(re.escape(marker) + r"(?:\r?\n)?" + rf"(?:{legacy_pattern.pattern})?")
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        replaced = False
        # Prefer replacing the combined marker block (machine marker + optional human-readable line)
        if combined_pattern.search(content) is not None:
            content = combined_pattern.sub(text, content, count=1)
            replaced = True
        else:
            # Fallback: replace whichever single marker exists
            if marker in content:
                content = content.replace(marker, text, 1)
                replaced = True
            elif legacy_pattern.search(content) is not None:
                content = legacy_pattern.sub(text, content, count=1)
                replaced = True
        if not replaced:
            return False
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content if content.endswith("\n") else (content + "\n"))
        return True
    except Exception as e:
        print(f"Warning: failed to replace marker for chunk {idx}: {e}")
        return False


def _clean_hidden_chain(text: str, tag: str) -> str:
    """隐藏思维链：删除例如 <think>...</think> 或自定义标签及其内部内容。

    参数:
    - text: 待清理文本
    - tag: 标签名（不带尖括号），如 "think"、"thinking"
    """
    if not tag:
        tag = "think"
    # 构造不区分大小写、允许属性/空白的匹配，最小匹配内部内容
    pattern = re.compile(rf"<\s*{re.escape(tag)}\b[^>]*>[\s\S]*?<\s*/\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    return re.sub(pattern, "", text)


def _clean_background_header(text: str) -> str:
    """清理模型偶发输出的“上下文/正文提示段”泄漏：
    1) 中文样式：『（以下内容仅做背景信息…不输出）……（以下内容是正文）』
    2) 英文样式：从 'Context (for reference only...)' 到 'Content to translate (OUTPUT ONLY...)' 的提示文本（含代码块）
    若未匹配到成对提示，保持原文。
    """
    s = text
    # 中文成对提示
    try:
        start_cn = r"[\(（]\s*以下内容.*?背景信息[^）\)]*不输出[^）\)]*[\)）]"
        end_cn = r"[\(（]\s*以下内容.*?正文[^）\)]*[\)）]"
        pattern_cn = re.compile(start_cn + r"[\s\S]*?" + end_cn, re.IGNORECASE)
        s = re.sub(pattern_cn, "", s)
    except Exception:
        pass
    # 英文提示：移除从 Context... 到 Content to translate... 标题行（含之间的代码围栏），仅保留标题后的内容
    try:
        start_en = re.compile(r"Context\s*\(for\s+reference\s+only[^)]*\)\s*:\s*", re.IGNORECASE)
        end_en = re.compile(r"Content\s+to\s+translate\s*\(OUTPUT\s+ONLY[^)]*\)\s*:\s*", re.IGNORECASE)
        m1 = start_en.search(s)
        m2 = end_en.search(s) if m1 else None
        if m1 and m2 and m2.start() > m1.start():
            # 删除 [m1.start(): m2.end()] 之间的文本（相当于去掉“Context…”到“Content to translate…”标题本身）
            s = s[:m1.start()] + s[m2.end():]
    except Exception:
        pass
    return s


def postprocess_output(text: str, *, hide_chain: bool = False, chain_tag: str = "think") -> str:
    """统一的输出后处理：
    1) 可选隐藏思维链 <tag>…</tag>
    2) 始终清理『（以下内容仅做背景信息，不输出）……（以下内容是正文）』泄漏
    """
    out = text or ""
    if hide_chain:
        out = _clean_hidden_chain(out, chain_tag or "think")
    out = _clean_background_header(out)
    return out


def append_text_to_output(output_path: str, text: str) -> None:
    with open(output_path, "a", encoding="utf-8") as out:
        out.write(text)
        if not text.endswith("\n"):
            out.write("\n")
        out.flush()


def write_failed_marker(output_path: str, idx: int, exc_type: str) -> None:
    marker = FAILED_MARKER_FMT.format(idx=idx)
    human = f"【第{idx}段翻译失败：{exc_type}】"
    append_text_to_output(output_path, f"{marker}\n{human}")


# ===== End helpers =====


def run_translation(
    input_path: str,
    output_path: str,
    base_url: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    max_chars: int,
    overlap_chars: int,
    context_only_overlap: bool,
    start_chunk: int,
    request_timeout: int,
    max_retries: int,
    skip_on_error: bool,
    use_stream: bool,
    # new controls
    auto_bisect_on_fail: bool = True,
    bisect_min_chars: int = 600,
    bisect_max_depth: int = 3,
    # newly added llm options
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 0.0,
    length_penalty: float = 0.0,
    # 输出后处理
    hide_chain: bool = False,
    chain_tag: str = "think",
    pdf_recognition: Optional[dict] = None,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    try:
        from document_loader import load_document
        text = load_document(input_path, pdf_recognition=pdf_recognition)
    except Exception as e:
        raise RuntimeError(f"Failed to load input document: {e}")

    sentences = split_sentences(text)

    if context_only_overlap:
        pairs = chunk_by_chars_pairs(sentences, max_chars=max_chars, overlap_chars=overlap_chars)
        total = len(pairs)
    else:
        chunks = chunk_by_chars(sentences, max_chars=max_chars, overlap_chars=overlap_chars)
        total = len(chunks)

    start_idx = max(1, int(start_chunk or 1))
    if start_idx > total:
        print(f"Start chunk {start_idx} is beyond total chunks {total}. Nothing to do.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        # Handle existing output based on flags
        if OVERWRITE_FLAG:
            open(output_path, "w", encoding="utf-8").close()
        elif RESUME_FLAG:
            pass  # append to existing file
        else:
            print(f"Output file already exists: {output_path}")
            resp = input("Overwrite? (y/N): ").strip().lower()
            if resp != "y":
                print("Aborted by user.")
                return
            open(output_path, "w", encoding="utf-8").close()

    try:
        from openai import OpenAI
    except Exception as _e:
        raise RuntimeError("OpenAI client is not available. Please install 'openai'.")
    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"Total chunks: {total}. Starting translation using model '{model}' at {base_url}... (resuming from chunk {start_idx})")

    def process_and_write(idx: int, ctx: str, content: str) -> None:
        print(f"Translating chunk {idx}/{total} (content_chars={len(content)}, ctx_chars={len(ctx)})...")
        try:
            out_text = try_translate_unit(
                client, model, source_lang, target_lang, ctx, content,
                request_timeout, max_retries, use_stream, idx,
                system_prompt=system_prompt,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            )
        except Exception as e:
            print(f"Primary translation failed for chunk {idx}: {type(e).__name__}: {e}")
            out_text = None
            if auto_bisect_on_fail:
                print(f"Attempting auto-bisect for chunk {idx}...")
                out_text = translate_with_bisect(
                    client, model, source_lang, target_lang, ctx, content,
                    request_timeout, max_retries, use_stream, idx,
                    bisect_min_chars, bisect_max_depth, 0,
                    system_prompt=system_prompt,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty, length_penalty=length_penalty,
                )
            if out_text is None:
                if skip_on_error:
                    write_failed_marker(output_path, idx, type(e).__name__)
                    print(f"Chunk {idx}/{total} marked as FAILED and recorded in output (auto-skip). Proceeding to next chunk.")
                    return
                else:
                    choice = input(f"Chunk {idx} still failed after bisect. Skip this segment? (y/N): ").strip().lower()
                    if choice == "y":
                        write_failed_marker(output_path, idx, type(e).__name__)
                        print(f"Chunk {idx}/{total} marked as FAILED and recorded in output. Proceeding to next chunk.")
                        return
                    else:
                        raise
        # If we reach here, we have out_text
        # 后处理：隐藏思维链与背景标记清理
        out_text = postprocess_output(out_text, hide_chain=hide_chain, chain_tag=chain_tag)
        # If previous run recorded a failed marker for this chunk, replace it in place; otherwise append
        if output_contains_failed_marker(output_path, idx):
            if replace_failed_marker_in_output(output_path, idx, out_text):
                print(f"Chunk {idx}/{total} done. Replaced previous failure marker in output.")
            else:
                append_text_to_output(output_path, out_text)
                print(f"Chunk {idx}/{total} done. Appended content translation to output (no marker found despite expectation).")
        else:
            append_text_to_output(output_path, out_text)
            print(f"Chunk {idx}/{total} done. Appended content translation to output.")

    if context_only_overlap:
        for idx, (ctx, content) in enumerate(pairs, start=1):
            if idx < start_idx:
                continue
            process_and_write(idx, ctx, content)
    else:
        for idx, chunk in enumerate(chunks, start=1):
            if idx < start_idx:
                continue
            # In non-context mode, use empty context
            process_and_write(idx, "", chunk)

    print(f"Translation completed. Output written to: {output_path}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunked translation via LM Studio OpenAI-compatible API"
    )
    parser.add_argument("--input", default=CONFIG["INPUT"], help="Path to input text file")
    parser.add_argument(
        "--output", default=CONFIG["OUTPUT"], help="Path to output file for translated text"
    )
    parser.add_argument(
        "--base-url",
        default=CONFIG["BASE_URL"],
        help="Base URL of the LM Studio server (OpenAI-compatible)",
    )
    parser.add_argument(
        "--api-key",
        default=CONFIG["API_KEY"],
        help="API key for the server (LM Studio usually accepts any non-empty string)",
    )
    parser.add_argument("--model", default=CONFIG["MODEL"], help="Model name as exposed by LM Studio")
    parser.add_argument(
        "--source-lang", default=CONFIG["SOURCE_LANG"], help="Source language code or name (e.g., 'auto', 'en')"
    )
    parser.add_argument(
        "--target-lang", default=CONFIG["TARGET_LANG"], help="Target language code or name (e.g., 'zh', 'en')"
    )
    parser.add_argument("--chunk-size-chars", type=int, default=CONFIG["CHUNK_SIZE_CHARS"], help="Max characters per chunk")
    parser.add_argument(
        "--overlap-chars", type=int, default=CONFIG["OVERLAP_CHARS"], help="Characters of overlap between chunks"
    )
    # Boolean options with defaults sourced from CONFIG (supports --overwrite/--no-overwrite)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["OVERWRITE"],
        help="Overwrite output file if it exists",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["RESUME"],
        help="Append to existing output file if it exists",
    )
    parser.add_argument(
        "--context-only-overlap",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["CONTEXT_ONLY_OVERLAP"],
        help="Use overlap only as context (do not translate or write it)",
    )
    parser.add_argument(
        "--start-chunk",
        type=int,
        default=CONFIG["START_CHUNK"],
        help="1-based chunk index to start from (useful for manual resume)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=CONFIG["REQUEST_TIMEOUT"],
        help="Per-request timeout in seconds to avoid hanging on long responses",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=CONFIG["MAX_RETRIES"],
        help="Max retries per chunk on error/timeout",
    )
    parser.add_argument(
        "--skip-on-error",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["SKIP_ON_ERROR"],
        help="Skip a chunk after retries exhausted (write a failure marker)",
    )
    parser.add_argument(
        "--use-stream",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["USE_STREAM"],
        help="Use streaming responses to reduce risk of hanging",
    )
    # New CLI for auto-bisect
    parser.add_argument(
        "--auto-bisect",
        action=argparse.BooleanOptionalAction,
        default=CONFIG["AUTO_BISECT_ON_FAIL"],
        help="Auto-bisect a failed chunk into smaller sub-chunks and retry",
    )
    parser.add_argument(
        "--bisect-min-chars",
        type=int,
        default=CONFIG["BISECT_MIN_CHARS"],
        help="Do not bisect further if content length is <= this",
    )
    parser.add_argument(
        "--bisect-max-depth",
        type=int,
        default=CONFIG["BISECT_MAX_DEPTH"],
        help="Maximum recursive bisect depth",
    )
    # New: system prompt & hyperparameters
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("LMSTUDIO_SYSTEM_PROMPT", ""),
        help="Custom system prompt for the translator role (overrides default)",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Path to a text file containing the system prompt; overrides --system-prompt if provided",
    )
    parser.add_argument("--temperature", type=float, default=CONFIG.get("TEMPERATURE", 0.0), help="Sampling temperature (0.0-2.0, 0 for deterministic)")
    parser.add_argument("--top-p", type=float, default=CONFIG.get("TOP_P", 1.0), help="Top-p nucleus sampling (0-1)")
    parser.add_argument("--top-k", type=int, default=CONFIG.get("TOP_K", 0), help="Top-k sampling (vendor-specific, 0 means use default)")
    parser.add_argument("--repetition-penalty", type=float, default=CONFIG.get("REPETITION_PENALTY", 0.0), help="Repetition penalty (vendor-specific, 0 to disable)")
    parser.add_argument("--length-penalty", type=float, default=CONFIG.get("LENGTH_PENALTY", 0.0), help="Length penalty (vendor-specific, 0 to disable)")
    # 输出后处理：隐藏思维链与自定义标签
    parser.add_argument(
        "--hide-chain",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.get("HIDE_CHAIN", False),
        help="Hide chain-of-thought blocks like <think>...</think> from outputs",
    )
    parser.add_argument(
        "--chain-tag",
        default=CONFIG.get("CHAIN_TAG", "think"),
        help="Tag name used to wrap chain-of-thought (e.g., 'think' => <think></think>)",
    )
    # PDF recognition
    parser.add_argument("--pdf-recognizer", choices=["auto","vlm","none"], default=CONFIG.get("PDF_RECOGNIZER","auto"), help="PDF recognition mode")
    parser.add_argument("--vlm-url", default=CONFIG.get("VLM_URL",""), help="VLM service URL")
    parser.add_argument("--vlm-key", default=CONFIG.get("VLM_KEY",""), help="VLM service key")
    parser.add_argument("--pdf-dpi", type=int, default=CONFIG.get("PDF_DPI",200), help="PDF render DPI")
    parser.add_argument("--pdf-pages", default=CONFIG.get("PDF_PAGES",""), help="PDF pages spec, e.g. '1-3,7'")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    # Resolve system prompt precedence
    sys_prompt_val: Optional[str] = None
    sp_file = getattr(ns, "system_prompt_file", None)
    try:
        if sp_file:
            with open(sp_file, "r", encoding="utf-8") as f:
                sys_prompt_val = f.read()
        else:
            sp_inline = getattr(ns, "system_prompt", "")
            sys_prompt_val = sp_inline if sp_inline else None
    except Exception as _e:
        print(f"Warning: failed to read system prompt file: {_e}")
        sys_prompt_val = None
    # Expose flags to run_translation scope via globals
    OVERWRITE_FLAG = bool(getattr(ns, "overwrite", False))
    RESUME_FLAG = bool(getattr(ns, "resume", False))
    try:
        run_translation(
            input_path=ns.input,
            output_path=ns.output,
            base_url=ns.base_url,
            api_key=ns.api_key,
            model=ns.model,
            source_lang=ns.source_lang,
            target_lang=ns.target_lang,
            max_chars=ns.chunk_size_chars,
            overlap_chars=ns.overlap_chars,
            context_only_overlap=bool(getattr(ns, "context_only_overlap", False)),
            start_chunk=int(getattr(ns, "start_chunk", 1)),
            request_timeout=int(getattr(ns, "request_timeout", CONFIG["REQUEST_TIMEOUT"])),
            max_retries=int(getattr(ns, "max_retries", CONFIG["MAX_RETRIES"])),
            skip_on_error=bool(getattr(ns, "skip_on_error", CONFIG["SKIP_ON_ERROR"])),
            use_stream=bool(getattr(ns, "use_stream", CONFIG["USE_STREAM"])),
            auto_bisect_on_fail=bool(getattr(ns, "auto_bisect", CONFIG["AUTO_BISECT_ON_FAIL"])),
            bisect_min_chars=int(getattr(ns, "bisect_min_chars", CONFIG["BISECT_MIN_CHARS"])),
            bisect_max_depth=int(getattr(ns, "bisect_max_depth", CONFIG["BISECT_MAX_DEPTH"])),
            system_prompt=sys_prompt_val,
            temperature=float(getattr(ns, "temperature", 0.0)),
            top_p=float(getattr(ns, "top_p", 1.0)),
            top_k=int(getattr(ns, "top_k", 0)),
            repetition_penalty=float(getattr(ns, "repetition_penalty", 0.0)),
            length_penalty=float(getattr(ns, "length_penalty", 0.0)),
            hide_chain=bool(getattr(ns, "hide_chain", CONFIG.get("HIDE_CHAIN", False))),
            chain_tag=str(getattr(ns, "chain_tag", CONFIG.get("CHAIN_TAG", "think")) or "think"),
            pdf_recognition={
                "mode": getattr(ns, "pdf_recognizer", CONFIG.get("PDF_RECOGNIZER","auto")),
                "vlm_url": getattr(ns, "vlm_url", CONFIG.get("VLM_URL","")),
                "vlm_key": getattr(ns, "vlm_key", CONFIG.get("VLM_KEY","")),
                "dpi": int(getattr(ns, "pdf_dpi", CONFIG.get("PDF_DPI",200))),
                "pages": getattr(ns, "pdf_pages", CONFIG.get("PDF_PAGES","")) or None,
            },
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
