import argparse
import os
import re
import sys
import time
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print(
        "The 'openai' package is required. Please install it first, e.g.:\n  uv pip install --python .venv\\Scripts\\python openai\n",
        file=sys.stderr,
    )
    sys.exit(1)

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


def build_messages(source_lang: str, target_lang: str, chunk: str) -> list:
    system_prompt = (
        f"You are a professional translator. Translate the user's text from {source_lang or 'the original language'} "
        f"to {target_lang}. Preserve original formatting, markdown structure, and line breaks. "
        f"Do not add explanations or comments. If there are code blocks or inline code, keep them unchanged."
    )
    user_prompt = "Translate the following content. Output only the translation.\n\n" + chunk
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_with_context(source_lang: str, target_lang: str, context: str, content: str) -> list:
    """Build chat messages where 'context' is for understanding only and must not be translated/output."""
    system_prompt = (
        f"You are a professional translator. Translate only the text in the 'Content' section from {source_lang or 'the original language'} "
        f"to {target_lang}. Use the 'Context' section only for understanding. Do NOT translate or include the Context in the output. "
        f"Preserve formatting, markdown structure, and line breaks in the translation."
    )
    user_prompt = (
        "Here is additional context and the content to translate.\n\n"
        "Context (for reference only, DO NOT translate or output):\n" +
        (f"```\n{context}\n```\n\n" if context else "(none)\n\n") +
        "Content to translate (OUTPUT ONLY the translation of this section):\n" +
        f"```\n{content}\n```"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def translate_chunk(
    client: OpenAI,
    model: str,
    messages: list,
    retries: int = 3,
    retry_delay: float = 2.0,
    timeout: int | float | None = None,
    use_stream: bool = True,
    skip_on_error: bool = False,
    chunk_index: int | None = None,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if use_stream:
                acc = []
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    stream=True,
                    timeout=timeout,
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
                    temperature=0,
                    timeout=timeout,
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
    client: OpenAI,
    model: str,
    source_lang: str,
    target_lang: str,
    context: str,
    content: str,
    request_timeout: int,
    max_retries: int,
    use_stream: bool,
    chunk_index: int,
) -> str:
    messages = build_messages_with_context(source_lang, target_lang, context, content)
    return translate_chunk(
        client, model, messages,
        retries=max_retries, timeout=request_timeout, use_stream=use_stream,
        skip_on_error=False, chunk_index=chunk_index,
    )


def translate_with_bisect(
    client: OpenAI,
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
        )
    except Exception:
        # bisect left further
        left_out = translate_with_bisect(
            client, model, source_lang, target_lang, context, left,
            request_timeout, max_retries, use_stream, chunk_index,
            min_chars, max_depth, depth + 1,
        )
    try:
        right_out = try_translate_unit(
            client, model, source_lang, target_lang, context, right,
            request_timeout, max_retries, use_stream, chunk_index,
        )
    except Exception:
        right_out = translate_with_bisect(
            client, model, source_lang, target_lang, context, right,
            request_timeout, max_retries, use_stream, chunk_index,
            min_chars, max_depth, depth + 1,
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
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

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

    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"Total chunks: {total}. Starting translation using model '{model}' at {base_url}... (resuming from chunk {start_idx})")

    def process_and_write(idx: int, ctx: str, content: str) -> None:
        print(f"Translating chunk {idx}/{total} (content_chars={len(content)}, ctx_chars={len(ctx)})...")
        try:
            out_text = try_translate_unit(
                client, model, source_lang, target_lang, ctx, content,
                request_timeout, max_retries, use_stream, idx,
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
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
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)