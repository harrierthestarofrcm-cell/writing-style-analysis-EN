# -*- coding: utf-8 -*-
from text_cleaning import clean_text

"""从每篇论文中提取可解释的写作特征"""
import re
import json
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np

# ============ 论文目录 ============
ESSAY_DIR = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\all years essay")

# ============ 输出目录：改到 essay_ethics_2023_2024_2025 下面 ============
OUTPUT_DIR = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\style_analysis\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".txt", ".md", ".docx", ".pdf"}

# ---------- 词表 ----------
STOPWORDS = set("""a an the and or but if while of to in on at by for with about as is are was were be been being
this that these those it its he she they them his her their we our you your i me my mine
not no nor so than then too very can could may might must shall should will would do does did done have has had""".split())

CONNECTIVES = set("""however therefore moreover furthermore nevertheless consequently thus hence
although though whereas because since while meanwhile additionally similarly likewise conversely
first second third finally lastly overall in conclusion for example for instance such as""".split())

CLAUSE_MARKERS = set("""that which who whom whose where when why how because although though
if unless until since while as before after""".split())

FIRST_PERSON = set("i me my mine we us our ours".split())
SECOND_PERSON = set("you your yours".split())
MODALS = set("may might could should would can will must shall".split())


# ---------- 截断参考文献区 ----------
_REF_SECTION_RE = re.compile(
    r'(?im)^[ \t]*(references|bibliography|works\s+cited|参考文献)[ \t]*$'
)


def remove_references_section(text: str) -> str:
    """截断正文后面的 References / Bibliography / Works Cited 部分"""
    m = _REF_SECTION_RE.search(text)
    if m:
        return text[:m.start()]
    return text


def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return ""
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    return ""


def syllables(word: str) -> int:
    """近似音节数，用于可读性公式"""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def extract_features(text: str) -> Dict[str, float]:
    # 分词
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = len(words)
    if n_words == 0:
        return {}

    # 分句（按 . ! ? 切）
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sents = max(1, len(sentences))
    sent_lengths = [len(re.findall(r"[A-Za-z']+", s)) for s in sentences]
    sent_lengths = [l for l in sent_lengths if l > 0] or [0]

    # 段落
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    n_paras = max(1, len(paragraphs))

    # 词汇
    unique_words = set(words)
    content_words = [w for w in words if w not in STOPWORDS]
    word_freq = Counter(words)
    rare_threshold = 1
    rare_words = [w for w, c in word_freq.items() if c <= rare_threshold and w not in STOPWORDS]

    # 词长
    avg_word_len = np.mean([len(w) for w in words])
    long_word_ratio = sum(1 for w in words if len(w) >= 7) / n_words

    # 句法
    avg_sent_len = np.mean(sent_lengths)
    std_sent_len = np.std(sent_lengths)
    long_sent_ratio = sum(1 for l in sent_lengths if l > 25) / n_sents
    short_sent_ratio = sum(1 for l in sent_lengths if l < 10) / n_sents
    clause_density = sum(1 for w in words if w in CLAUSE_MARKERS) / n_words

    # 语篇
    connective_freq = sum(1 for w in words if w in CONNECTIVES) / n_words
    avg_para_words = n_words / n_paras
    sent_per_para = n_sents / n_paras

    # 可读性（Flesch Reading Ease 近似）
    total_syll = sum(syllables(w) for w in words)
    asl = n_words / n_sents
    asw = total_syll / n_words
    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    fk_grade = 0.39 * asl + 11.8 * asw - 15.59

    # 标点
    def pcount(pat): return len(re.findall(pat, text))
    comma_freq = pcount(r",") / n_words
    semi_freq = pcount(r";") / n_words
    dash_freq = pcount(r"[-—]") / n_words
    paren_freq = pcount(r"[()]") / n_words
    qmark_freq = pcount(r"\?") / n_words
    excl_freq = pcount(r"!") / n_words
    quote_freq = pcount(r'["""]') / n_words

    # 语气
    first_person = sum(1 for w in words if w in FIRST_PERSON) / n_words
    second_person = sum(1 for w in words if w in SECOND_PERSON) / n_words
    modals = sum(1 for w in words if w in MODALS) / n_words

    return {
        # 基础
        "n_words": n_words,
        "n_sentences": n_sents,
        "n_paragraphs": n_paras,
        # 词汇
        "type_token_ratio": len(unique_words) / n_words,
        "content_word_ratio": len(content_words) / n_words,
        "rare_word_ratio": len(rare_words) / n_words,
        "avg_word_length": float(avg_word_len),
        "long_word_ratio": float(long_word_ratio),
        # 句法
        "avg_sentence_length": float(avg_sent_len),
        "std_sentence_length": float(std_sent_len),
        "long_sentence_ratio": float(long_sent_ratio),
        "short_sentence_ratio": float(short_sent_ratio),
        "clause_marker_density": float(clause_density),
        # 语篇
        "connective_density": float(connective_freq),
        "avg_paragraph_length": float(avg_para_words),
        "sentences_per_paragraph": float(sent_per_para),
        # 可读性
        "flesch_reading_ease": float(flesch),
        "flesch_kincaid_grade": float(fk_grade),
        # 标点
        "comma_freq": float(comma_freq),
        "semicolon_freq": float(semi_freq),
        "dash_freq": float(dash_freq),
        "parenthesis_freq": float(paren_freq),
        "question_freq": float(qmark_freq),
        "exclamation_freq": float(excl_freq),
        "quote_freq": float(quote_freq),
        # 语气
        "first_person_ratio": float(first_person),
        "second_person_ratio": float(second_person),
        "modal_density": float(modals),
    }


def extract_student_id(path: Path, root: Path) -> str:
    """从文件名最后一段提取学生 ID"""
    stem = path.stem
    if "IndividualEssay_" in stem:
        return stem.split("IndividualEssay_")[-1]
    tokens = re.split(r"[_\-\\s]+", stem)
    return tokens[-1] if tokens else stem


def main():
    files = [p for p in ESSAY_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    print(f"共 {len(files)} 篇论文")

    paper_features = []
    for path in files:
        text = read_text(path)
        if not text.strip():
            continue

        text = clean_text(text, drop_refs=True, drop_urls=True)
        # 截断参考文献区（双保险）
        text = remove_references_section(text)

        if len(text.split()) < 100:   # 清洗后太短，跳过
            print(f"[跳过] {path.name}: 清洗后内容过短")
            continue

        feats = extract_features(text)
        if not feats:
            continue
        feats["paper_file"] = path.name
        feats["student_id"] = extract_student_id(path, ESSAY_DIR)
        paper_features.append(feats)

    # 按学生聚合（取均值）
    by_student: Dict[str, List[Dict]] = {}
    for f in paper_features:
        by_student.setdefault(f["student_id"], []).append(f)

    numeric_keys = [k for k in paper_features[0] if k not in ("paper_file", "student_id")]

    student_features = {}
    for sid, feats_list in by_student.items():
        agg = {"n_papers": len(feats_list)}
        for k in numeric_keys:
            agg[k] = float(np.mean([f[k] for f in feats_list]))
        agg["paper_files"] = [f["paper_file"] for f in feats_list]
        student_features[sid] = agg

    # 保存
    with open(OUTPUT_DIR / "paper_features.json", "w", encoding="utf-8") as f:
        json.dump(paper_features, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "student_features.json", "w", encoding="utf-8") as f:
        json.dump(student_features, f, ensure_ascii=False, indent=2)

    # 打印每个学生的风格摘要
    print("\n" + "=" * 70)
    print("学生写作风格特征摘要")
    print("=" * 70)
    highlight = [
        "type_token_ratio", "avg_sentence_length", "std_sentence_length",
        "connective_density", "first_person_ratio", "modal_density",
        "flesch_reading_ease", "comma_freq",
    ]
    for sid, agg in student_features.items():
        print(f"\n【{sid}】 ({agg['n_papers']} 篇)")
        for k in highlight:
            print(f"  {k:28s} {agg[k]:.3f}")
        print(f"  论文: {agg['paper_files']}")

    print(f"\n详细结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()