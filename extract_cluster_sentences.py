# -*- coding: utf-8 -*-
"""
为每个 cluster 提取代表性示例

根据 cluster 类型自适应选择提取单位：
  - 段落型 cluster（特征涉及 avg_paragraph_length / sentences_per_paragraph）
        → 提取代表性段落（同时附上该段落内的关键句）
  - 句子型 cluster（其他）
        → 提取代表性句子

两种单位都用 cluster 的 z-score 画像打分，并做去重、引用过滤、乱码过滤。
每个 cluster 输出详细文字解释。
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict
from difflib import SequenceMatcher

from text_cleaning import clean_text, is_citation_sentence

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


# ================================================================
# 去重工具
# ================================================================
def normalize_sentence(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def is_near_duplicate(a_norm: str, b_norm: str,
                      ratio_thresh: float = 0.90,
                      contain_thresh: float = 0.85) -> bool:
    if not a_norm or not b_norm:
        return True
    if a_norm == b_norm:
        return True
    short, long_ = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)
    if len(short) >= 15 and short in long_ and len(short) / len(long_) >= contain_thresh:
        return True
    if min(len(a_norm), len(b_norm)) / max(len(a_norm), len(b_norm)) < 0.6:
        return False
    return SequenceMatcher(None, a_norm, b_norm).ratio() >= ratio_thresh


# ================================================================
# 参考文献 / 乱码 / 截断过滤
# ================================================================
_REF_SECTION_RE = re.compile(
    r'(?im)^[ \t]*(references|bibliography|works\s+cited|参考文献)[ \t]*$'
)


def remove_references_section(text: str) -> str:
    m = _REF_SECTION_RE.search(text)
    if m:
        return text[:m.start()]
    return text


_REF_PATTERNS = [
    re.compile(r'^\s*\[\d+\]'),
    re.compile(r'^\s*\d+\.\s'),
    re.compile(r'\bAvailable\s*:', re.I),
    re.compile(r'\baccessed\b', re.I),
    re.compile(r'\bRetrieved\s+from\b', re.I),
    re.compile(r'\bdoi\s*:', re.I),
    re.compile(r'https?://', re.I),
    re.compile(r'www\.', re.I),
    re.compile(r'\bet\s+al\.', re.I),
    re.compile(r'\bRefence\b', re.I),
    re.compile(r'\bReference\s*:', re.I),
    re.compile(r'\bProceedings\s+of\b', re.I),
    re.compile(r'\bJournal\s+of\b', re.I),
    re.compile(r'\bConference\s+on\b', re.I),
    re.compile(r'\bSpringer\b'),
    re.compile(r'\bIEEE\b'),
    re.compile(r'\bACM\b'),
    re.compile(r'\barXiv\b', re.I),
    re.compile(r'\bISBN\b'),
    re.compile(r'\bISSN\b'),
    re.compile(r'\bVol\.\s*\d+', re.I),
    re.compile(r'\bNo\.\s*\d+', re.I),
    re.compile(r'\bpp\.\s*\d+', re.I),
    re.compile(r'\bChapter\s+\d+', re.I),
    re.compile(r'\bArt\.\s*\d+', re.I),
    re.compile(r'\(\s*\d{4}\s*\)'),
    re.compile(r'\b(19|20)\d{2}\b'),
    re.compile(r'\b[A-Z][a-z]+,\s*[A-Z]\.'),
    re.compile(r'[A-Z][a-z]+\s+["“][^"”]+["”]\s*,'),
    re.compile(r'%[0-9A-F]{2}'),
    re.compile(
        r'\b(Intersoft|Investopedia|Medium|GoDaddy|WITNESS|Harvard\s+Business\s+Review|'
        r'Yale\s+Daily\s+News|The\s+Table\s+Read|JAMA\s+Network\s+Open|'
        r'Global\s+Market\s+Insights|GDPR\.eu|Solidity\s+by\s+example|'
        r'Content\s+Authenticity\s+Initiative|OECD\s+AI\s+Principles|'
        r'European\s+Parliament|HKIE|Solidity)\b', re.I),
    re.compile(r'^[A-Z][a-z]+\s*,\s+[A-Z][a-z]+'),
]


def is_reference_like(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 20:
        return True
    for pat in _REF_PATTERNS:
        if pat.search(s):
            return True
    digit_ratio = len(re.findall(r'[0-9]', s)) / max(1, len(s))
    punct_ratio = len(re.findall(r'[^\w\s]', s)) / max(1, len(s))
    weird_mix = re.search(r'[A-Za-z]\d[A-Za-z]|\d[A-Za-z]{2,}\d', s)
    if digit_ratio > 0.10 and punct_ratio > 0.12:
        return True
    if punct_ratio > 0.30:
        return True
    if weird_mix:
        return True
    return False


def is_garbled_text(sent: str) -> bool:
    s = sent.strip()
    if len(s) < 25:
        return False
    words = re.findall(r'[A-Za-z]+', s)
    if not words:
        return True
    if any(len(w) >= 18 for w in words):
        return True
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 8.5:
        return True
    very_long = sum(1 for w in words if len(w) > 11)
    if very_long / len(words) > 0.25:
        return True
    space_ratio = s.count(' ') / len(s)
    if space_ratio < 0.08:
        return True
    return False


def is_truncated_sentence(sent: str) -> bool:
    s = sent.strip()
    if not s:
        return True
    last = s[-1]
    if last not in '.!?"\')]”’':
        return True
    if last in ',;:':
        return True
    return False


def is_bad_sentence(sent: str) -> bool:
    """统一的坏句判断"""
    if not sent or not sent.strip():
        return True
    try:
        if is_citation_sentence(sent):
            return True
    except NameError:
        pass
    if is_reference_like(sent):
        return True
    if is_garbled_text(sent):
        return True
    if is_truncated_sentence(sent):
        return True
    return False


# ============ 配置 ============
# 输出目录改到 essay_ethics_2023_2024_2025 下面
OUT = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\style_analysis\outputs")
TOP_N_SENTENCES = 10
TOP_N_PARAGRAPHS = 5
MIN_SENT_WORDS = 5
MIN_PARA_SENTENCES = 2        # 段落至少几句
MIN_PARA_WORDS = 40           # 段落至少几词
MAX_PARA_DISPLAY_WORDS = 500  # 段落显示时的截断上限

# ============ 词表 ============
CONNECTIVES = set("""however therefore moreover furthermore nevertheless consequently thus hence
although though whereas because since while meanwhile additionally similarly likewise conversely
first second third finally lastly overall in conclusion for example for instance such as""".split())

CLAUSE_MARKERS = set("""that which who whom whose where when why how because although though
if unless until since while as before after""".split())

FIRST_PERSON = set("i me my mine we us our ours".split())
SECOND_PERSON = set("you your yours".split())
MODALS = set("may might could should would can will must shall".split())


# ================================================================
# Cluster 解释模块
# ================================================================
FEATURE_DESCRIPTIONS = {
    "type_token_ratio":        ("词汇多样性",       "用词重复度较高",         "用词丰富多样"),
    "avg_word_length":         ("平均词长",         "偏向短词、日常用语",     "偏向长词、正式用语"),
    "long_word_ratio":         ("长词比例",         "长词较少",               "长词较多"),
    "rare_word_ratio":         ("罕见词比例",       "用词常见",               "用词独特"),
    "avg_sentence_length":     ("平均句长",         "句子偏短",               "句子偏长"),
    "std_sentence_length":     ("句长变化",         "句长稳定均匀",           "句长起伏大"),
    "long_sentence_ratio":     ("长句比例",         "长句较少",               "长句较多"),
    "short_sentence_ratio":    ("短句比例",         "短句较少",               "短句较多"),
    "clause_marker_density":   ("从句密度",         "从句少、句式线性",       "从句多、句式复杂"),
    "connective_density":      ("连接词密度",       "连接词少、逻辑直接",     "连接词多、逻辑显式"),
    "first_person_ratio":      ("第一人称",         "客观、第三人称化",       "主观、第一人称化"),
    "second_person_ratio":     ("第二人称",         "不与读者对话",           "直接对话读者"),
    "modal_density":           ("情态动词密度",     "语气果断、断言强",       "语气谨慎、留余地"),
    "comma_freq":              ("逗号频率",         "停顿少",                 "停顿频繁"),
    "semicolon_freq":          ("分号频率",         "几乎不用分号",           "常用分号"),
    "dash_freq":               ("破折号频率",       "几乎不用破折号",         "常用破折号"),
    "parenthesis_freq":        ("括号频率",         "几乎不用括号",           "常用括号补充"),
    "question_freq":           ("问句频率",         "几乎不问句",             "常用问句"),
    "exclamation_freq":        ("感叹号频率",       "几乎不用感叹号",         "常用感叹号"),
    "flesch_reading_ease":     ("可读性(FRES)",     "偏难读",                 "易读"),
    "flesch_kincaid_grade":    ("可读年级难度",     "偏简单",                 "偏难"),
    "avg_paragraph_length":    ("段落平均长度",     "段落偏短",               "段落偏长"),
    "sentences_per_paragraph": ("每段句子数",       "段落内句子少",           "段落内句子多"),
}


def _cluster_distance(prof_a, prof_b):
    keys = list(prof_a["z_scores"].keys())
    a = np.array([prof_a["z_scores"][k] for k in keys])
    b = np.array([prof_b["z_scores"][k] for k in keys])
    return float(np.linalg.norm(a - b))


PARAGRAPH_FEATURES = {"avg_paragraph_length", "sentences_per_paragraph"}


def decide_extraction_mode(prof, top_k=5):
    """
    判断这个 cluster 应该提取段落还是句子。
    返回 (mode, trigger_features)
      mode: "paragraph" 或 "sentence"
      trigger_features: 触发段落提取的特征名集合
    """
    z = prof["z_scores"]
    top_keys = [k for k, _ in sorted(z.items(), key=lambda x: -abs(x[1]))[:top_k]]
    overlap = set(top_keys) & PARAGRAPH_FEATURES
    if overlap:
        return "paragraph", overlap
    return "sentence", set()


def generate_cluster_explanation(cid, prof, all_profiles=None):
    z = prof["z_scores"]
    name = prof["auto_name"]
    n = prof["n_students"]

    valid_keys = [k for k in z if k in FEATURE_DESCRIPTIONS]
    sorted_items = sorted([(k, z[k]) for k in valid_keys], key=lambda x: -abs(x[1]))
    top5 = sorted_items[:5]

    mode, trigger = decide_extraction_mode(prof)

    L = []
    L.append("┌" + "─" * 86)
    L.append(f"│ Cluster {cid} 说明：{name}")
    L.append(f"│ 学生数：{n}")
    L.append(f"│ 提取模式：{'段落型（提取代表性段落）' if mode == 'paragraph' else '句子型（提取代表性句子）'}")
    if trigger:
        L.append(f"│ 触发段落提取的特征：{', '.join(sorted(trigger))}")
    L.append("└" + "─" * 86)
    L.append("")

    L.append("【为什么这些学生会被归为一类】")
    L.append("  聚类算法：KMeans，K=8")
    L.append("  特征空间：23 维可解释写作特征（词汇 / 句法 / 语篇 / 标点 / 语气）")
    L.append("  标准化方式：先对每个特征做 StandardScaler（z-score），再裁剪到 [-3, 3]")
    L.append("  分类依据：算法把 23 维空间中彼此靠近的学生分到同一个 cluster。")
    L.append("")

    L.append("【这个 cluster 最突出的 5 个维度】")
    L.append("  （z 值 = 相对全体学生的标准差倍数；正值=高于平均，负值=低于平均）")
    L.append("")
    for k, v in top5:
        if k not in FEATURE_DESCRIPTIONS:
            continue
        cn, low, high = FEATURE_DESCRIPTIONS[k]
        sign = "+" if v >= 0 else "-"
        direction = "高于平均" if v > 0 else "低于平均"
        meaning = high if v > 0 else low
        L.append(f"    {sign}  {cn:12s}  z = {v:+.2f}   {direction} → {meaning}")
    L.append("")

    L.append("【风格画像（自然语言总结）】")
    high_k = [k for k, v in top5 if v > 0]
    low_k  = [k for k, v in top5 if v < 0]
    high_cn = [FEATURE_DESCRIPTIONS[k][0] for k in high_k]
    low_cn  = [FEATURE_DESCRIPTIONS[k][0] for k in low_k]
    if high_cn:
        L.append(f"  显著高于平均：{'、'.join(high_cn)}")
    if low_cn:
        L.append(f"  显著低于平均：{'、'.join(low_cn)}")
    L.append("")

    top5_map = {k: (1 if v > 0 else -1) for k, v in top5}
    has = lambda k, s: top5_map.get(k) == s

    if has("avg_paragraph_length", +1) or has("sentences_per_paragraph", +1):
        L.append("  → 段落偏长、每段包含多句：句子累积成段，信息密度高。")
    if has("avg_paragraph_length", -1) or has("sentences_per_paragraph", -1):
        L.append("  → 段落偏短、每段句子少：节奏紧凑，段落边界频繁。")
    if has("connective_density", +1):
        L.append("  → 行文中连接词（however, therefore, such as 等）使用较多，逻辑推进显式。")
    if has("connective_density", -1):
        L.append("  → 行文中连接词较少，句子之间靠语义自然过渡。")
    if has("avg_sentence_length", +1):
        L.append("  → 句子整体偏长，倾向在单句中承载多层信息。")
    if has("avg_sentence_length", -1):
        L.append("  → 句子整体偏短，节奏干脆，信息逐条列出。")
    if has("first_person_ratio", +1):
        L.append("  → 常以第一人称（we / our / I）叙述，语气带个人视角。")
    if has("first_person_ratio", -1):
        L.append("  → 极少使用第一人称，叙述偏客观、第三人称化。")
    if has("second_person_ratio", +1):
        L.append("  → 常用第二人称（you / your）直接对话读者。")
    if has("modal_density", +1):
        L.append("  → 情态动词（should / may / could）使用较多，语气谨慎、留有余地。")
    if has("modal_density", -1):
        L.append("  → 情态动词少，语气果断、断言性强。")
    if has("long_word_ratio", +1) or has("avg_word_length", +1):
        L.append("  → 用词偏正式、偏向长词和专业术语。")
    if has("long_word_ratio", -1) or has("avg_word_length", -1):
        L.append("  → 用词偏通俗、偏向日常短词，可读性较强。")
    if has("parenthesis_freq", +1):
        L.append("  → 常用括号补充说明（缩写、定义、示例等），信息密度高。")
    if has("dash_freq", +1):
        L.append("  → 常用破折号插入补充，节奏带停顿感。")
    if has("question_freq", +1):
        L.append("  → 常用问句，语气像在与读者对话。")
    if has("clause_marker_density", +1):
        L.append("  → 从句标记（that / which / because 等）使用多，句式偏复杂。")
    if has("clause_marker_density", -1):
        L.append("  → 从句标记少，句式偏线性、简单。")
    L.append("")

    # 与其他 cluster 的对比
    if all_profiles:
        others = [(int(c), p) for c, p in all_profiles.items()
                  if int(c) != cid and isinstance(p, dict) and "z_scores" in p]
        if others:
            dists = [(c, _cluster_distance(prof, p)) for c, p in others]
            dists.sort(key=lambda x: x[1])
            near_c, near_d = dists[0]
            far_c, far_d = dists[-1]
            near_name = all_profiles[str(near_c)]["auto_name"]
            far_name  = all_profiles[str(far_c)]["auto_name"]
            L.append("【与其他 cluster 的关系】")
            L.append(f"  最接近的 cluster：C{near_c}（{near_name}），距离 {near_d:.2f}")
            L.append(f"  最远的 cluster：  C{far_c}（{far_name}），距离 {far_d:.2f}")
            L.append("")

    # 判分说明（根据 mode 切换）
    L.append("【代表示例是如何被选出来的】")
    if mode == "paragraph":
        L.append("  提取单位：段落（因为该 cluster 的画像与段落长度/句数强相关）")
        L.append("  1. 按空行或句子聚合，把每篇论文切成若干段落。")
        L.append("  2. 对每个段落提取 16 维段落级特征：")
        L.append("       段落词数、段落句数、段落平均句长、段落平均词长、长词比例、")
        L.append("       连接词密度、从句密度、人称、情态、各种标点频率。")
        L.append("  3. 在全体段落中标准化为 z-score。")
        L.append("  4. 段落得分 = 段落 z 向量 · cluster z 向量（点积）。")
        L.append("  5. 取 Top 5 段落，并附上该段内得分最高的关键句。")
    else:
        L.append("  提取单位：句子（因为该 cluster 的画像与句子级特征强相关）")
        L.append("  1. 对全文每个句子提取 14 维句子级特征。")
        L.append("  2. 在全体句子中标准化为 z-score。")
        L.append("  3. 句子得分 = 句子 z 向量 · cluster z 向量（点积）。")
        L.append("  4. 取 Top 10 句子，并做近重复去重。")
        L.append("  5. 每条句子下面列出贡献最大的 3–4 个维度。")
    L.append("")
    return "\n".join(L)


# ================================================================
# 句子级特征
# ================================================================
_BRAND_EXCEPTIONS = {
    "osu!", "yahoo!", "jeopardy!", "air!", "wow!", "help!",
    "e!", "oh!", "aha!", "hey!", "hi!", "ok!", "yes!", "no!",
}

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "cf", "al", "fig", "no", "vol", "pp", "ed", "eds",
    "inc", "ltd", "co", "corp", "dept", "univ", "approx", "est",
}

_NEXT_SENT_START = r'["\'(\[\u201c\u2018A-Z0-9]'


def sentence_features(sent: str) -> Dict[str, float]:
    for brand in _BRAND_EXCEPTIONS:
        sent = re.sub(re.escape(brand), brand.rstrip("!?"), sent, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z']+", sent.lower())
    n = max(1, len(words))
    n_conn    = sum(1 for w in words if w in CONNECTIVES)
    n_clause  = sum(1 for w in words if w in CLAUSE_MARKERS)
    n_first   = sum(1 for w in words if w in FIRST_PERSON)
    n_second  = sum(1 for w in words if w in SECOND_PERSON)
    n_modal   = sum(1 for w in words if w in MODALS)
    n_comma   = sent.count(",")
    n_semi    = sent.count(";")
    n_dash    = len(re.findall(r"[-—]", sent))
    n_paren   = len(re.findall(r"[()]", sent))
    n_q       = sent.count("?")
    n_excl    = sent.count("!")
    avg_wlen  = float(np.mean([len(w) for w in words])) if words else 0.0
    long_r    = sum(1 for w in words if len(w) >= 7) / n
    return {
        "avg_sentence_length":    float(n),
        "avg_word_length":        avg_wlen,
        "long_word_ratio":        long_r,
        "connective_density":     n_conn / n,
        "clause_marker_density":  n_clause / n,
        "first_person_ratio":     n_first / n,
        "second_person_ratio":    n_second / n,
        "modal_density":          n_modal / n,
        "comma_freq":             n_comma / n,
        "semicolon_freq":         n_semi / n,
        "dash_freq":              n_dash / n,
        "parenthesis_freq":       n_paren / n,
        "question_freq":          n_q / n,
        "exclamation_freq":       n_excl / n,
    }


# ================================================================
# 段落级特征
# ================================================================
PARA_FEAT_NAMES = [
    "avg_paragraph_length",
    "sentences_per_paragraph",
    "avg_sentence_length",
    "avg_word_length",
    "long_word_ratio",
    "connective_density",
    "clause_marker_density",
    "first_person_ratio",
    "second_person_ratio",
    "modal_density",
    "comma_freq",
    "semicolon_freq",
    "dash_freq",
    "parenthesis_freq",
    "question_freq",
    "exclamation_freq",
]


def paragraph_features(para: str) -> Dict[str, float]:
    sents = split_sentences(para, min_words=3)
    n_sents = max(1, len(sents))
    words = re.findall(r"[A-Za-z']+", para.lower())
    n_words = max(1, len(words))

    if sents:
        sfeats = [sentence_features(s) for s in sents]
        avg = {k: float(np.mean([f[k] for f in sfeats])) for k in sfeats[0]}
    else:
        avg = {k: 0.0 for k in PARA_FEAT_NAMES if k not in
               ("avg_paragraph_length", "sentences_per_paragraph")}

    return {
        "avg_paragraph_length":    float(n_words),
        "sentences_per_paragraph": float(n_sents),
        **avg,
    }


# ================================================================
# 文本读取
# ================================================================
def read_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码: {path}")


def read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError(f"不支持: {ext}")


# ================================================================
# 分句 / 分段
# ================================================================
def _protect_tokens(text: str):
    protected = []
    def _stash(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"
    text = re.sub(r"https?://\S+|www\.\S+", _stash, text)
    text = re.sub(r"\b10\.\d{4,9}/\S+", _stash, text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", _stash, text)
    text = re.sub(r"\b\d+\.\d+(\.\d+)*\b", _stash, text)
    text = re.sub(
        r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
        _stash, text, flags=re.IGNORECASE,
    )
    for brand in _BRAND_EXCEPTIONS:
        text = re.sub(re.escape(brand), _stash, text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", _stash, text)
    return text, protected


def _restore_tokens(text: str, protected: list) -> str:
    for i, tok in enumerate(protected):
        text = text.replace(f"\x00{i}\x00", tok)
    return text


def split_sentences(text: str, min_words: int = 5) -> List[str]:
    try:
        text = clean_text(text, drop_refs=True, drop_urls=True)
    except NameError:
        pass
    text = remove_references_section(text)
    text = re.sub(r'(?<=[a-z]{2})(?=[A-Z][a-z]{2})', ' ', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text, protected = _protect_tokens(text)
    pattern = re.compile(r'(?<=[.!?])\s+(?=' + _NEXT_SENT_START + r')')
    raw = pattern.split(text)
    raw = [_restore_tokens(s, protected) for s in raw]

    out = []
    for s in raw:
        s = s.strip().replace("\n", " ")
        s = re.sub(r"\s+", " ", s)
        if len(re.findall(r"[A-Za-z']+", s)) < min_words:
            continue
        if is_bad_sentence(s):
            continue
        out.append(s)
    return out


def split_paragraphs(text: str, fallback_per_para: int = 5) -> List[str]:
    """
    段落切分：
      1) 优先按空行切
      2) 如果切出来的段落数太少（<3），说明原文没有空行分隔，
         回退到"每 fallback_per_para 句聚合为一段"
    段落会过滤掉包含坏句的部分。
    """
    try:
        text = clean_text(text, drop_refs=True, drop_urls=True)
    except NameError:
        pass
    text = remove_references_section(text)

    # 按空行切
    raw_paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    paras = []
    for p in raw_paras:
        # 段内分句并过滤坏句
        sents = split_sentences(p, min_words=MIN_SENT_WORDS)
        if len(sents) >= MIN_PARA_SENTENCES:
            joined = " ".join(sents)
            if len(joined.split()) >= MIN_PARA_WORDS:
                paras.append(joined)

    # 回退：按句子聚合
    if len(paras) < 3:
        all_sents = split_sentences(text, min_words=MIN_SENT_WORDS)
        paras = []
        for i in range(0, len(all_sents), fallback_per_para):
            group = all_sents[i:i + fallback_per_para]
            if len(group) >= MIN_PARA_SENTENCES:
                joined = " ".join(group)
                if len(joined.split()) >= MIN_PARA_WORDS:
                    paras.append(joined)
    return paras


# ================================================================
# 加载数据
# ================================================================
print("加载 metadata / cluster 画像 ...")
with open(OUT / "metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
paper_meta = meta["paper_meta"]

with open(OUT / "cluster_profiles_feature_based.json", encoding="utf-8") as f:
    cluster_profiles = json.load(f)

student_to_cluster = {}
for cid_str, prof in cluster_profiles.items():
    for sid in prof["students"]:
        student_to_cluster[sid] = int(cid_str)

# ================================================================
# 预提取：句子 + 段落
# ================================================================
print("读取所有论文，切句 / 切段 ...")

paper_sent_cache: Dict[str, List[str]] = {}
paper_para_cache: Dict[str, List[str]] = {}


def get_sentences(paper_path, paper_file):
    if paper_file in paper_sent_cache:
        return paper_sent_cache[paper_file]
    try:
        sents = split_sentences(load_text(Path(paper_path)), min_words=MIN_SENT_WORDS)
    except Exception as e:
        print(f"  [警告-句] {paper_file}: {e}")
        sents = []
    paper_sent_cache[paper_file] = sents
    return sents


def get_paragraphs(paper_path, paper_file):
    if paper_file in paper_para_cache:
        return paper_para_cache[paper_file]
    try:
        paras = split_paragraphs(load_text(Path(paper_path)))
    except Exception as e:
        print(f"  [警告-段] {paper_file}: {e}")
        paras = []
    paper_para_cache[paper_file] = paras
    return paras


# 句子特征表
all_sent_feats = []
all_sent_info = []
# 段落特征表
all_para_feats = []
all_para_info = []

for p in paper_meta:
    sid = p["student_id"]
    # 句子
    for s in get_sentences(p["paper_path"], p["paper_file"]):
        all_sent_feats.append(sentence_features(s))
        all_sent_info.append((sid, p["paper_file"], s))
    # 段落
    for para in get_paragraphs(p["paper_path"], p["paper_file"]):
        all_para_feats.append(paragraph_features(para))
        all_para_info.append((sid, p["paper_file"], para))

print(f"  句子总数: {len(all_sent_info)}")
print(f"  段落总数: {len(all_para_info)}")

if not all_sent_info:
    print("[错误] 没有可用句子。")
    raise SystemExit(1)

# ================================================================
# 特征标准化
# ================================================================
SENT_FEAT_NAMES = list(all_sent_feats[0].keys())
S = np.array([[f[k] for k in SENT_FEAT_NAMES] for f in all_sent_feats])
mu_s, sd_s = S.mean(0), S.std(0) + 1e-9
S_z = np.clip((S - mu_s) / sd_s, -3, 3)

if all_para_feats:
    P = np.array([[f[k] for k in PARA_FEAT_NAMES] for f in all_para_feats])
    mu_p, sd_p = P.mean(0), P.std(0) + 1e-9
    P_z = np.clip((P - mu_p) / sd_p, -3, 3)
else:
    P_z = None

# ================================================================
# 每个 cluster 找代表示例
# ================================================================
results = {}

for cid_str, prof in cluster_profiles.items():
    cid = int(cid_str)
    z_scores = prof["z_scores"]
    mode, trigger = decide_extraction_mode(prof)

    explanation = generate_cluster_explanation(cid, prof, cluster_profiles)

    examples = []
    extraction_mode = mode  # 实际执行的模式

    if mode == "paragraph" and P_z is not None:
        # ---------- 段落提取 ----------
        cluster_vec = np.zeros(len(PARA_FEAT_NAMES))
        for j, k in enumerate(PARA_FEAT_NAMES):
            if k in z_scores:
                cluster_vec[j] = z_scores[k]

        scores = P_z @ cluster_vec
        mask = np.array([student_to_cluster.get(sid) == cid
                         for sid, _, _ in all_para_info])
        masked_scores = np.where(mask, scores, -np.inf)

        if not np.any(np.isfinite(masked_scores)):
            # 该 cluster 没有对应学生的段落 → 回退到句子
            extraction_mode = "sentence (回退：无段落样本)"
        else:
            order = np.argsort(-np.where(np.isfinite(masked_scores),
                                          masked_scores, -np.inf))
            selected_norms = []
            for i in order:
                if not np.isfinite(masked_scores[i]):
                    continue
                sid, paper_file, para = all_para_info[i]
                norm = normalize_sentence(para)
                if any(is_near_duplicate(norm, prev) for prev in selected_norms):
                    continue
                selected_norms.append(norm)

                # 段落里找一句关键句
                para_sents = split_sentences(para, min_words=MIN_SENT_WORDS)
                key_sent = None
                if para_sents:
                    pf = [sentence_features(s) for s in para_sents]
                    sf = np.array([[f[k] for k in SENT_FEAT_NAMES] for f in pf])
                    sf_z = np.clip((sf - mu_s) / sd_s, -3, 3)
                    sv = np.zeros(len(SENT_FEAT_NAMES))
                    for j, k in enumerate(SENT_FEAT_NAMES):
                        if k in z_scores:
                            sv[j] = z_scores[k]
                    ss = sf_z @ sv
                    key_sent = para_sents[int(np.argmax(ss))]

                # 段落贡献特征（用段落向量）
                contrib = P_z[i] * cluster_vec
                top_feat_idx = np.argsort(-np.abs(contrib))[:4]
                top_feats = [(PARA_FEAT_NAMES[j], float(contrib[j]))
                             for j in top_feat_idx if abs(contrib[j]) > 0.15]

                examples.append({
                    "rank": len(examples) + 1,
                    "score": float(scores[i]),
                    "student_id": sid,
                    "paper_file": paper_file,
                    "type": "paragraph",
                    "n_words": len(para.split()),
                    "n_sentences": len(para_sents),
                    "paragraph": para,
                    "key_sentence": key_sent,
                    "top_contributing_features": top_feats,
                })

                if len(examples) >= TOP_N_PARAGRAPHS:
                    break

    # 句子型（或回退）
    if not examples:
        extraction_mode = "sentence"
        cluster_vec = np.zeros(len(SENT_FEAT_NAMES))
        for j, k in enumerate(SENT_FEAT_NAMES):
            if k in z_scores:
                cluster_vec[j] = z_scores[k]

        scores = S_z @ cluster_vec
        mask = np.array([student_to_cluster.get(sid) == cid
                         for sid, _, _ in all_sent_info])
        masked_scores = np.where(mask, scores, -np.inf)

        if np.any(np.isfinite(masked_scores)):
            order = np.argsort(-np.where(np.isfinite(masked_scores),
                                          masked_scores, -np.inf))
            selected_norms = []
            for i in order:
                if not np.isfinite(masked_scores[i]):
                    continue
                sid, paper_file, sent = all_sent_info[i]
                norm = normalize_sentence(sent)
                if any(is_near_duplicate(norm, prev) for prev in selected_norms):
                    continue
                selected_norms.append(norm)

                contrib = S_z[i] * cluster_vec
                top_feat_idx = np.argsort(-np.abs(contrib))[:4]
                top_feats = [(SENT_FEAT_NAMES[j], float(contrib[j]))
                             for j in top_feat_idx if abs(contrib[j]) > 0.15]

                examples.append({
                    "rank": len(examples) + 1,
                    "score": float(scores[i]),
                    "student_id": sid,
                    "paper_file": paper_file,
                    "type": "sentence",
                    "sentence": sent,
                    "top_contributing_features": top_feats,
                })
                if len(examples) >= TOP_N_SENTENCES:
                    break

    results[cid] = {
        "cluster_id": cid,
        "cluster_name": prof["auto_name"],
        "n_students": prof["n_students"],
        "extraction_mode": extraction_mode,
        "trigger_features": sorted(list(trigger)),
        "defining_features": sorted(
            [(k, v) for k, v in z_scores.items() if k in SENT_FEAT_NAMES],
            key=lambda x: -abs(x[1])
        )[:5],
        "explanation": explanation,
        "examples": examples,
    }

    # ---------- 终端打印 ----------
    print("\n" + explanation)
    print("-" * 90)
    if examples and examples[0].get("type") == "paragraph":
        print(f"代表性段落（共 {len(examples)} 段）:")
    else:
        print(f"代表性句子（共 {len(examples)} 句）:")
    for ex in examples:
        if ex.get("type") == "paragraph":
            print(f"\n[段落 {ex['rank']} | score={ex['score']:.2f}] "
                  f"{ex['student_id']} / {ex['paper_file']}  "
                  f"({ex['n_words']} words, {ex['n_sentences']} sentences)")
            snippet = ex["paragraph"]
            if len(snippet.split()) > MAX_PARA_DISPLAY_WORDS:
                snippet = " ".join(snippet.split()[:MAX_PARA_DISPLAY_WORDS]) + " ..."
            print(f"  {snippet}")
            if ex.get("key_sentence"):
                print(f"  关键句: {ex['key_sentence']}")
            if ex["top_contributing_features"]:
                print("  贡献特征: " + ", ".join(
                    f"{k}({v:+.2f})" for k, v in ex["top_contributing_features"]))
        else:
            print(f"\n[Rank {ex['rank']} | score={ex['score']:.2f}] "
                  f"{ex['student_id']} / {ex['paper_file']}")
            print(f"  句子: {ex['sentence'][:400]}")
            if ex["top_contributing_features"]:
                print("  贡献特征: " + ", ".join(
                    f"{k}({v:+.2f})" for k, v in ex["top_contributing_features"]))

# ================================================================
# 保存 JSON
# ================================================================
out_json = OUT / "cluster_sentence_examples.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n保存: {out_json}")

# ================================================================
# 保存 TXT
# ================================================================
out_txt = OUT / "cluster_sentence_examples.txt"
with open(out_txt, "w", encoding="utf-8") as f:
    for cid in sorted(results.keys()):
        r = results[cid]
        f.write("=" * 90 + "\n")
        f.write(f"Cluster {cid}: {r['cluster_name']}  (n_students={r['n_students']})\n")
        f.write(f"提取模式: {r['extraction_mode']}\n")
        f.write("=" * 90 + "\n\n")

        f.write(r["explanation"])
        f.write("\n")

        f.write("-" * 90 + "\n")
        if r["examples"] and r["examples"][0].get("type") == "paragraph":
            f.write("代表性段落\n")
        else:
            f.write("代表性句子\n")
        f.write("-" * 90 + "\n\n")

        if not r["examples"]:
            f.write("（该 cluster 没有可用的代表示例）\n\n")
        for ex in r["examples"]:
            if ex.get("type") == "paragraph":
                f.write(f"[段落 {ex['rank']} | score={ex['score']:.2f}] "
                        f"{ex['student_id']} / {ex['paper_file']}  "
                        f"({ex['n_words']} words, {ex['n_sentences']} sentences)\n")
                f.write(f"段落:\n{ex['paragraph']}\n")
                if ex.get("key_sentence"):
                    f.write(f"关键句: {ex['key_sentence']}\n")
                if ex["top_contributing_features"]:
                    f.write("贡献特征: " + ", ".join(
                        f"{k}({v:+.2f})" for k, v in ex["top_contributing_features"]) + "\n")
                f.write("\n")
            else:
                f.write(f"[Rank {ex['rank']} | score={ex['score']:.2f}] "
                        f"{ex['student_id']} / {ex['paper_file']}\n")
                f.write(f"句子: {ex['sentence']}\n")
                if ex["top_contributing_features"]:
                    f.write("贡献特征: " + ", ".join(
                        f"{k}({v:+.2f})" for k, v in ex["top_contributing_features"]) + "\n")
                f.write("\n")
        f.write("\n\n")

print(f"保存: {out_txt}")