#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
from collections import defaultdict

# ========== 路径已写死 ==========
INPUT_DIR  = r"C:\Users\yuton\OneDrive\桌面\2024"
OUTPUT_DIR = r"C:\Users\yuton\OneDrive\桌面\2024_cleaned"
# ================================


# ---------- 读取 ----------
def read_docx(path):
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_pdf(path):
    import fitz
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def read_text(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_any(path):
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    return read_text(path)


# ---------- 页眉行特征 ----------
def is_header_signature(s):
    """判断一行是否含有页眉特征的字段值。"""
    s = s.strip()
    if not s:
        return False
    # 姓名：CHOI, Yin Fei / LI, Chung Him
    if re.match(r'^[A-Z][A-Z\s]*,\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?$', s):
        return True
    # 学号 7-9 位数字
    if re.match(r'^\d{7,9}$', s):
        return True
    # 项目代码：SYQ1 / DES4 / HO3 / GCH2
    if re.match(r'^[A-Z]{2,6}\d{1,4}$', s):
        return True
    # 日期：18th October, 2024 / 15/10/2024
    if re.match(r'^\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s*,?\s*\d{4}$', s, re.I):
        return True
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', s):
        return True
    return False


def remove_repeating_blocks(text, min_block=3, max_block=12):
    """
    删除文档中逐字重复出现的短行块。
    页眉/页脚会在每一页重复出现，因此会命中。
    """
    lines = text.split('\n')
    n = len(lines)
    norm = [ln.strip() for ln in lines]

    def is_short(s):
        return bool(s) and len(s) < 120 and not re.search(r'[.?!]\s*$', s)

    short = [is_short(s) for s in norm]
    to_delete = set()

    for length in range(min_block, max_block + 1):
        block_pos = defaultdict(list)
        for i in range(n - length + 1):
            # 块内所有行都必须是短行
            if not all(short[i:i + length]):
                continue
            block = tuple(norm[i:i + length])
            if not any(block):
                continue
            block_pos[block].append(i)

        for block, positions in block_pos.items():
            if len(positions) < 2:
                continue
            # 块内至少要有一行是页眉特征
            if not any(is_header_signature(s) for s in block):
                continue
            for start in positions:
                for k in range(start, start + length):
                    to_delete.add(k)

    return '\n'.join(lines[i] for i in range(n) if i not in to_delete)


# ---------- 常量 ----------
LABELS = [
    "Name", "Student ID", "Project Code", "Project Title",
    "Essay Title", "Date", "Word Count", "Total Pages"
]

BODY_START_MARKER = re.compile(
    r'Write\s+your\s+(?:reflection|analytical\s+essay)\s+below',
    re.I
)

SECTION_HEADER_RE = re.compile(
    r'^\s*#{0,6}\s*'
    r'(Overview|Introduction|Intro|Abstract|Conclusion|'
    r'System\s+Distribution|Collection\s+and\s+Use\s+of\s+User\s+Data|'
    r'Data\s+Contamination|Ethical\s+Issue\s*\d*|'
    r'Issue\s*\d*\s*[:：]?.*|Problem|Solution|Review|'
    r'Background|Methodology|Discussion|Analysis)\b',
    re.I
)


# ---------- 逐行判断 ----------
def is_metadata_line(s):
    s = s.strip()
    if not s:
        return False
    if s in LABELS:
        return True
    if 'FYP/FYT Essay on Professional' in s:
        return True
    if 'Professional, Ethical, Legal, Security, Social' in s:
        return True
    if 'Issues and Responsibilities' in s:
        return True
    if 'Please remove Page' in s and 'FYPMS' in s:
        return True
    if re.search(r'</?table|</?tr|</?td|colspan', s, re.I):
        return True
    hits = sum(1 for k in LABELS if k in s)
    if hits >= 2:
        return True
    if re.search(r'Name(?=[A-Z])', s):
        return True
    if re.search(r'Student\s*ID\s*\d', s):
        return True
    if re.search(r'Project\s*Code[A-Z0-9]', s):
        return True
    if re.search(r'Project\s*Title[A-Z]', s):
        return True
    if re.search(r'Essay\s*Title[A-Z]', s):
        return True
    if re.search(r'Word\s*Count\s*\d', s):
        return True
    if re.search(r'Total\s*Pages\s*\d', s):
        return True
    if re.search(r'Date\s*\d', s):
        return True
    if re.search(r'ITSC\s*a/', s, re.I):
        return True
    if re.match(r'^ITSC\s', s, re.I):
        return True
    return False


def is_noise_line(s):
    s = s.strip()
    if not s:
        return False
    if re.match(r'^\s*(?:[-–—]\s*)?Page\s*\d+\s*(?:of\s*\d+)?(?:\s*[-–—])?\s*$', s, re.I):
        return True
    if re.match(r'^\s*\d+\s*/\s*\d+\s*$', s):
        return True
    if re.match(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', s):
        return True
    if re.match(r'^\s*\d{1,3}\s*$', s):
        return True
    if re.match(r'^\s*20\d{2}\s*$', s):
        return True
    if re.match(r'^\s*[\d\s]{40,}\s*$', s):
        if len(re.findall(r'\d', s)) >= 20:
            return True
    return False


def is_body_like(s):
    if len(s) > 120:
        return True
    if re.search(r'[.?!]\s+\w', s):
        return True
    if len(s.split()) >= 8:
        return True
    return False


# ---------- 定位正文起点 ----------
def find_body_start(lines):
    for i, line in enumerate(lines):
        if BODY_START_MARKER.search(line):
            return i + 1
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if is_metadata_line(s):
            continue
        if is_noise_line(s):
            continue
        if SECTION_HEADER_RE.match(s):
            return i
        if is_body_like(s):
            return i
    return 0


def clean_page(page_text):
    lines = page_text.split('\n')
    body_start = find_body_start(lines)
    lines = lines[body_start:]
    out = []
    for line in lines:
        s = line.strip()
        if is_noise_line(s):
            continue
        if is_metadata_line(s):
            continue
        out.append(line)
    return '\n'.join(out)


# ---------- 参考文献 ----------
REF_TITLE_RE       = re.compile(r'(?im)^\s*#{0,6}\s*References?\s*$')
REF_TITLE_COLON_RE = re.compile(r'(?im)^\s*References?\s*[:：]\s*$')

REF_PATTERNS_STRICT = [
    r'(?im)^\s*20\d{2}\.\s*\[Online\]',
    r'(?im)^\s*Available:\s*https?://',
    r'(?im)^\s*Accessed:\s*',
    r'(?im)^\s*doi:\s*10\.',
    r'(?im)^\s*\[\d+\]\s+[A-Z][a-z]+,\s+[A-Z]',
    r'(?im)^\s*\d+\)\s+[A-Z][a-z]+,\s+[A-Z]',
]


def find_reference_start(text):
    m = REF_TITLE_RE.search(text) or REF_TITLE_COLON_RE.search(text)
    if m:
        return m.start()
    half = len(text) // 2
    search_text = text[half:]
    cut = None
    for pat in REF_PATTERNS_STRICT:
        for mm in re.finditer(pat, search_text):
            pos = half + mm.start()
            if cut is None or pos < cut:
                cut = pos
    return cut


# ---------- 主清理 ----------
def clean_essay_text(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 步骤 1：删除重复出现的页眉块（每页都一样的短行块）
    text = remove_repeating_blocks(text)

    # 步骤 2：按页拆分并清理每页
    parts = re.split(r'=====\s*Page\s+\d+\s*=====\s*\n?', text)
    parts = [clean_page(p) for p in parts]
    text = '\n'.join(parts)

    # 步骤 3：截断参考文献
    cut = find_reference_start(text)
    if cut is not None:
        text = text[:cut]

    # 步骤 4：压缩空行
    lines = [ln.strip() for ln in text.split('\n')]
    out = []
    prev_empty = False
    for ln in lines:
        if ln == '':
            if not prev_empty:
                out.append('')
            prev_empty = True
        else:
            out.append(ln)
            prev_empty = False
    while out and out[0] == '':
        out.pop(0)
    while out and out[-1] == '':
        out.pop()
    return '\n'.join(out).strip()


# ---------- 批量处理 ----------
def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.pdf', '.docx', '.txt', '.md'}
    files = [p for p in input_dir.rglob('*') if p.suffix.lower() in exts]

    print(f'输入: {input_dir}')
    print(f'输出: {output_dir}')
    print(f'找到 {len(files)} 个文件\n')

    ok, err = 0, 0
    for i, path in enumerate(files, 1):
        try:
            raw = read_any(path)
            cleaned = clean_essay_text(raw)
            out_path = output_dir / f'{path.stem}_clean.txt'
            out_path.write_text(cleaned, encoding='utf-8')
            print(f'[{i}/{len(files)}] OK: {path.name}')
            ok += 1
        except Exception as e:
            print(f'[{i}/{len(files)}] FAIL: {path.name}: {e}')
            err += 1

    print(f'\n完成。成功 {ok}，失败 {err}')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()