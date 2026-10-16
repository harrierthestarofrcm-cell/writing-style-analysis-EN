#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 2024_cleaned 的基础上，重新整理段落、断句、换行。
输入: 2024_cleaned/*.txt
输出: 2024_final/*_final.txt
"""

import re
from pathlib import Path

# ========== 路径已写死 ==========
INPUT_DIR  = r"C:\Users\yuton\OneDrive\桌面\2024_cleaned"
OUTPUT_DIR = r"C:\Users\yuton\OneDrive\桌面\2024_final"
# ================================


# ---------- 章节标题模式 ----------
SECTION_PATTERNS = [
    r'^Overview\s*:?\s*$',
    r'^Introduction\s*:?\s*$',
    r'^Intro\s*:?\s*$',
    r'^Conclusion\s*:?\s*$',
    r'^Abstract\s*:?\s*$',
    r'^Summary\s*:?\s*$',
    r'^Issue\s+\d+\s*[:：].+',
    r'^Ethical\s+Issue\s+\d*\s*[:：].+',
    r'^System\s+Distribution\s*$',
    r'^Collection\s+and\s+Use\s+of\s+User\s+Data\s*$',
    r'^Data\s+Contamination\s*$',
    r'^Background\s*$',
    r'^Motivation\s*$',
    r'^Methodology\s*$',
    r'^Method\s*$',
    r'^Discussion\s*$',
    r'^Analysis\s*$',
    r'^Problem\s*$',
    r'^Solution\s*$',
    r'^Review\s*$',
    r'^Related\s+Work\s*$',
    r'^System\s+Design\s*$',
    r'^System\s+Architecture\s*$',
    r'^Implementation\s*$',
    r'^Evaluation\s*$',
    r'^Results?\s*$',
    r'^Findings\s*$',
    r'^Recommendations?\s*$',
    r'^Future\s+Work\s*$',
    r'^Limitations?\s*$',
]
SECTION_RE = re.compile('|'.join(SECTION_PATTERNS), re.I)

# 连字符断行时，如果 '-' 前最后一个词在这些词里，保留连字符
# 例如 "up-to-" + "date" → "up-to-date"（不是 "up-todate"）
# 而 "comput-" + "ing" → "computing"（不是 "comput-ing"）
KEEP_HYPHEN_WORDS = {
    'up', 'to', 'in', 'on', 'at', 'by', 'of', 'or', 'for', 'and', 'the',
    'a', 'an', 'with', 'from', 'into', 'over', 'under', 'well', 'self',
    'non', 'pre', 'post', 'anti', 'semi', 'multi', 'inter', 'intra',
    'co', 're', 'de', 'un', 'out', 'off', 'no', 'so', 'go', 'do',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
}


def looks_like_header(s: str) -> bool:
    return bool(SECTION_RE.match(s.strip()))


def fix_hyphen(buf: str) -> str:
    """处理行尾连字符：真复合词保留，断词去除。"""
    prefix = buf.rstrip()
    if not prefix.endswith('-'):
        return prefix
    body = prefix[:-1]
    last_word = re.split(r'\s+', body)[-1].lower()
    if last_word in KEEP_HYPHEN_WORDS:
        return prefix      # 保留连字符
    return body            # 删除连字符


def is_paragraph_break(prev: str, nxt: str) -> bool:
    """
    根据空行前后的内容，判断这个空行是否真的表示段落分隔。
      - 前一行以句末标点结尾，下一行以大写/数字/[ 开头 → 是段落分隔
      - 其他情况（例如 "...tattoos" 后面空行接 "[5]."） → 不分割，继续拼接
    """
    prev = prev.rstrip()
    nxt = nxt.lstrip()
    if not prev or not nxt:
        return False

    # 前一行必须以句末标点结尾
    if not re.search(r'[.?!]\s*["\'”’\)\]]*\s*$', prev):
        return False

    # 下一行以大写字母、数字或开头符号起始
    c = nxt[0]
    return c.isupper() or c.isdigit() or c in '[("\'#“'


def merge_paragraphs(lines):
    """把被 PDF 拆碎的行列表合并成段落。"""
    out = []
    buf = ""
    i = 0
    n = len(lines)

    while i < n:
        s = lines[i].strip()

        # ---- 空行 ----
        if not s:
            # 找下一个非空行
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1

            if buf:
                if j >= n:
                    # 文件结尾
                    out.append(buf)
                    buf = ""
                else:
                    nxt = lines[j].strip()
                    if is_paragraph_break(buf, nxt):
                        out.append(buf)
                        buf = ""
                    # 否则继续合并（空行只是 PDF 伪影）
            i += 1
            continue

        # ---- 章节标题 ----
        if looks_like_header(s):
            if buf:
                out.append(buf)
                buf = ""
            out.append(s)
            i += 1
            continue

        # ---- 拼接到当前段落 ----
        if not buf:
            buf = s
        elif buf.endswith('-'):
            buf = fix_hyphen(buf) + s
        else:
            buf = buf + ' ' + s
        i += 1

    if buf:
        out.append(buf)

    return out


def reformat(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    paragraphs = merge_paragraphs(lines)
    # 用空行连接段落
    return '\n\n'.join(paragraphs).strip()


# ---------- 批量处理 ----------
def process_file(path: Path, out_dir: Path) -> Path:
    raw = path.read_text(encoding='utf-8', errors='ignore')
    cleaned = reformat(raw)
    out_path = out_dir / f"{path.stem}_final.txt"
    out_path.write_text(cleaned, encoding='utf-8')
    return out_path


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob('*.txt'))
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(files)} 个 txt 文件\n")

    ok, err = 0, 0
    for i, path in enumerate(files, 1):
        try:
            out_path = process_file(path, output_dir)
            print(f"[{i}/{len(files)}] OK  : {path.name} -> {out_path.name}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(files)}] FAIL: {path.name}: {e}")
            err += 1

    print(f"\n完成。成功 {ok}，失败 {err}")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()