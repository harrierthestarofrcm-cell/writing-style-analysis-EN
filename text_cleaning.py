# -*- coding: utf-8 -*-
"""文本清洗：去除参考文献区、URL、DOI、邮箱、纯引用条目"""
import re

# ---------- URL / DOI / Email ----------
URL_PATTERN   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOI_PATTERN   = re.compile(r"\bdoi:\s*\S+|\b10\.\d{4,9}/\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# ---------- 参考文献区标题 ----------
# 支持 "References"、"Bibliography"、"Works Cited"、"5. References"、"VI. BIBLIOGRAPHY" 等
REF_HEADING_RE = re.compile(
    r"^\s*(?:[IVXLC]+\.|\d+\.)?\s*"
    r"(references|bibliography|works\s+cited|reference\s+list|"
    r"literature\s+cited|citations?|参考文献)\s*:?\s*$",
    re.IGNORECASE,
)


def strip_references_section(text: str) -> str:
    """
    找到最后一个参考文献标题，截断它及其后的所有内容。
    只匹配独立成行的短标题，避免误伤正文里的 'references' 一词。
    """
    lines = text.split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) > 80:
            continue
        if REF_HEADING_RE.match(s):
            cut = i
    return "\n".join(lines[:cut])


def remove_urls_dois_emails(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = DOI_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    return text


def clean_text(text: str,
               drop_refs: bool = True,
               drop_urls: bool = True) -> str:
    """主清洗入口；特征提取与句子提取都用它"""
    if drop_refs:
        text = strip_references_section(text)
    if drop_urls:
        text = remove_urls_dois_emails(text)
    return text


# ---------- 句子级判定 ----------
_AUTHOR_YEAR_RE = re.compile(r"^[A-Z][A-Za-z\-']+,\s*[A-Z]\.")
_YEAR_PAREN_RE  = re.compile(r"\(\d{4}[a-z]?\)\.")
_JOURNAL_RE     = re.compile(r"\b(vol|no|pp|ed|eds)\.?\s*\d", re.IGNORECASE)
_BRACKET_REF_RE = re.compile(r"^\[\d+\]")
_RETRIEVED_RE   = re.compile(r"^\s*(retrieved|accessed|available\s+(at|from)|"
                             r"from\s+https?|see\s+https?)", re.IGNORECASE)


def is_citation_sentence(sent: str) -> bool:
    """
    判断一句话是不是引用条目（而非正文）。
    返回 True 表示应丢弃。
    """
    s = sent.strip()
    if not s:
        return True

    # 含 URL / DOI / Email
    if URL_PATTERN.search(s) or DOI_PATTERN.search(s) or EMAIL_PATTERN.search(s):
        return True

    # 以 [1] 开头
    if _BRACKET_REF_RE.match(s):
        return True

    # 以 "Retrieved ..." / "Available at ..." 开头
    if _RETRIEVED_RE.match(s):
        return True

    # "Smith, J. (2020). ..." 这种参考文献格式
    if _AUTHOR_YEAR_RE.match(s) and _YEAR_PAREN_RE.search(s):
        return True

    # 年份 + 期刊号
    if _YEAR_PAREN_RE.search(s) and _JOURNAL_RE.search(s):
        return True

    # 非字母字符占比过高（URL、代码、DOI 片段）
    n_alpha = sum(1 for c in s if c.isalpha())
    if len(s) > 0 and n_alpha / len(s) < 0.45:
        return True

    return False