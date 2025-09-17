import re
import argparse
from typing import List, Tuple
import sys


def split_sentences(text: str):
    text = text.replace("\r\n", "\n")
    code_blocks = re.split(r"(```[\s\S]*?```)", text)
    sentences = []
    sentence_end_pattern = re.compile(r"(?<=[。！？!?.])\s+|\n+")
    for block in code_blocks:
        if block.startswith("```") and block.endswith("```"):
            sentences.append(block)
        else:
            parts = [p for p in re.split(sentence_end_pattern, block) if p and not p.isspace()]
            sentences.extend(parts)
    return sentences


def chunk_by_chars_pairs(sentences, max_chars: int, overlap_chars: int):
    pairs = []
    current_context = ""
    current_content = []
    current_len = len(current_context)
    for sent in sentences:
        s_len = len(sent)
        if current_content and current_len + s_len > max_chars:
            content_text = "".join(current_content).strip()
            if content_text:
                pairs.append((current_context, content_text))
            combined = current_context + content_text
            next_context = combined[-overlap_chars:] if overlap_chars > 0 else ""
            current_context = next_context
            current_content = []
            current_len = len(current_context)
        current_content.append(sent)
        current_len += s_len
    if current_content:
        content_text = "".join(current_content).strip()
        if content_text:
            pairs.append((current_context, content_text))
    return pairs


def analyze(label: str, s: str):
    punct = "。！？!?."
    chars = len(s)
    newlines = s.count("\n")
    codeblocks = s.count("```")
    longest = 0
    cur = 0
    for ch in s:
        if ch in punct or ch == "\n":
            if cur > longest:
                longest = cur
            cur = 0
        else:
            cur += 1
    if cur > longest:
        longest = cur
    head = s[:200].replace("\n", " ")
    tail = s[-200:].replace("\n", " ")
    print(f"[{label}] len={chars}, newlines={newlines}, codeblocks={codeblocks}, longest_no_punct_run={longest}")
    print(f"  head: {head}")
    print(f"  tail: {tail}")


# ===== New helpers for CLI and formatting =====

def parse_chunk_indices(spec: str, total: int) -> List[int]:
    """Parse chunk spec like '191,192,194' or '150-155,191'. Keep order and de-duplicate."""
    result: List[int] = []
    seen = set()
    tokens = [t.strip() for t in spec.split(',') if t.strip()]
    for t in tokens:
        if '-' in t:
            try:
                a, b = t.split('-', 1)
                start = int(a)
                end = int(b)
                if start <= end:
                    rng = range(start, end + 1)
                else:
                    rng = range(start, end - 1, -1)
                for i in rng:
                    if 1 <= i <= total and i not in seen:
                        seen.add(i)
                        result.append(i)
            except ValueError:
                continue
        else:
            try:
                i = int(t)
                if 1 <= i <= total and i not in seen:
                    seen.add(i)
                    result.append(i)
            except ValueError:
                continue
    return result


def build_chunk_report(idx: int, ctx: str, content: str) -> str:
    def stats_block(label: str, s: str) -> str:
        punct = "。！？!?."
        chars = len(s)
        newlines = s.count("\n")
        codeblocks = s.count("```")
        longest = 0
        cur = 0
        for ch in s:
            if ch in punct or ch == "\n":
                if cur > longest:
                    longest = cur
                cur = 0
            else:
                cur += 1
        if cur > longest:
            longest = cur
        head = s[:200].replace("\n", " ")
        tail = s[-200:].replace("\n", " ")
        return (
            f"[{label}] len={chars}, newlines={newlines}, codeblocks={codeblocks}, longest_no_punct_run={longest}\n"
            f"  head: {head}\n"
            f"  tail: {tail}\n"
        )

    lines = [
        f"== Chunk {idx} ==",
        stats_block("content", content).rstrip(),
        stats_block("context", ctx).rstrip(),
        "--- Context (DO NOT translate) ---",
        ctx,
        "--- Content (to translate) ---",
        content,
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect specific translation chunks with stats and content output")
    parser.add_argument("--input", default="whole.txt", help="Path to input text file")
    parser.add_argument("--chunks", default="191,192,194", help="Chunk indices, e.g. '191,192,194' or ranges '150-155,191'")
    parser.add_argument("--max-chars", type=int, default=3000, help="Max characters per chunk (should match translator)")
    parser.add_argument("--overlap", type=int, default=120, help="Overlap characters between chunks (should match translator)")
    parser.add_argument("--export", default=None, help="If provided, write the report to this txt file instead of printing")
    args = parser.parse_args()

    # Mitigate Windows console encoding issues (e.g., cp936) to avoid crashes when printing Unicode
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    path = args.input
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    max_chars = args.max_chars
    overlap = args.overlap

    sentences = split_sentences(text)
    pairs = chunk_by_chars_pairs(sentences, max_chars, overlap)
    total = len(pairs)

    indices = parse_chunk_indices(args.chunks, total)
    if not indices:
        print(f"No valid chunk indices parsed from '{args.chunks}'. Total available: {total}")
        return

    reports: List[str] = []
    for idx in indices:
        ctx, content = pairs[idx - 1]
        reports.append(build_chunk_report(idx, ctx, content))

    output_text = "\n".join(reports)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as out:
            out.write(output_text)
        print(f"Report written to: {args.export}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()