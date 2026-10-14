# -*- coding: utf-8 -*-
"""
学生论文写作风格嵌入提取流程
步骤：加载 → 分段 → 嵌入 → 聚合 → 保存 → 分析
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ============ 1. 配置 ============

# 论文目录
ESSAY_DIR = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\all years essay")

# 输出目录：改到 essay_ethics_2023_2024_2025 下面
OUTPUT_DIR = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\style_analysis\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# 分段参数（按词数近似 token 数；0.6B 模型最大 32k，这里留足余量）
CHUNK_SIZE_WORDS = 400      # 每段目标词数
CHUNK_OVERLAP_WORDS = 50    # 段间重叠，避免切碎语义
MIN_CHUNK_WORDS = 80        # 太短的段丢弃

# 批大小（显存小就调小，比如 4 或 8）
BATCH_SIZE = 16

# 支持的文件格式
SUPPORTED_EXTS = {".txt", ".md", ".docx", ".pdf"}

# ============ 2. 文本加载 ============


def read_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法解码文件: {path}")


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
    raise ValueError(f"不支持的文件类型: {ext}")


# ============ 3. 学生 ID 识别 ============

def extract_student_id(path: Path, root: Path) -> str:
    """从文件名最后一段提取学生 ID"""
    stem = path.stem
    if "IndividualEssay_" in stem:
        return stem.split("IndividualEssay_")[-1]
    tokens = re.split(r"[_\-\\s]+", stem)
    return tokens[-1] if tokens else stem


# ============ 4. 分段 ============

def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS,
               min_words: int = MIN_CHUNK_WORDS) -> List[str]:
    """
    按词切分（对英文友好；中文场景可改成按字符数切分）。
    先按段落聚合，再打包成固定大小的 chunk。
    """
    # 按空行切段
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf_len >= min_words:
            chunks.append(" ".join(buf))
        buf, buf_len = [], 0

    for para in paragraphs:
        words = para.split()
        if buf_len + len(words) > chunk_size and buf_len > 0:
            flush()
            # 重叠：从上一段末尾回取
            if chunks and overlap > 0:
                tail = chunks[-1].split()[-overlap:]
                buf = [" ".join(tail)]
                buf_len = len(tail)
        buf.append(para)
        buf_len += len(words)

    flush()
    return chunks


# ============ 5. 主流程 ============

def main():
    # ---- 5.1 收集文件 ----
    files: List[Path] = []
    for p in ESSAY_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)

    if not files:
        print(f"[警告] 在 {ESSAY_DIR} 下没找到支持的文件。")
        return

    print(f"共找到 {len(files)} 篇论文。")

    # ---- 5.2 加载 + 分段 ----
    all_chunks: List[str] = []
    chunk_meta: List[Dict] = []   # 每个 chunk 的元信息

    for path in tqdm(files, desc="读取并分段"):
        student_id = extract_student_id(path, ESSAY_DIR)
        try:
            text = load_text(path)
        except Exception as e:
            print(f"[跳过] {path.name}: {e}")
            continue

        chunks = chunk_text(text)
        if not chunks:
            print(f"[跳过] {path.name}: 分段后为空")
            continue

        for i, c in enumerate(chunks):
            all_chunks.append(c)
            chunk_meta.append({
                "student_id": student_id,
                "paper_file": path.name,
                "paper_path": str(path),
                "chunk_index": i,
                "n_words": len(c.split()),
            })

    print(f"共生成 {len(all_chunks)} 个文本块。")

    # ---- 5.3 加载模型 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    # 缩短最大长度以省显存（默认可到 32k，太长没必要）
    model.max_seq_length = 1024

    # ---- 5.4 生成 chunk 嵌入 ----
    print("开始生成 chunk 嵌入 ...")
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"chunk 嵌入形状: {chunk_embeddings.shape}")

    # ---- 5.5 聚合到 paper 级别 ----
    # 每个 paper 的嵌入 = 其所有 chunk 嵌入的平均
    paper_to_chunks: Dict[str, List[int]] = {}
    for idx, meta in enumerate(chunk_meta):
        key = meta["paper_path"]
        paper_to_chunks.setdefault(key, []).append(idx)

    paper_embeddings = []
    paper_meta = []
    for paper_path, idxs in paper_to_chunks.items():
        vec = chunk_embeddings[idxs].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)  # 归一化
        paper_embeddings.append(vec)
        paper_meta.append({
            "paper_path": paper_path,
            "paper_file": Path(paper_path).name,
            "student_id": chunk_meta[idxs[0]]["student_id"],
            "n_chunks": len(idxs),
        })
    paper_embeddings = np.stack(paper_embeddings)

    # ---- 5.6 聚合到 student 级别（写作风格画像）----
    student_to_papers: Dict[str, List[int]] = {}
    for idx, meta in enumerate(paper_meta):
        student_to_papers.setdefault(meta["student_id"], []).append(idx)

    student_embeddings = []
    student_meta = []
    for sid, idxs in student_to_papers.items():
        vec = paper_embeddings[idxs].mean(axis=0)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        student_embeddings.append(vec)
        student_meta.append({
            "student_id": sid,
            "n_papers": len(idxs),
            "paper_files": [paper_meta[i]["paper_file"] for i in idxs],
        })
    student_embeddings = np.stack(student_embeddings)

    # ---- 5.7 保存 ----
    np.save(OUTPUT_DIR / "chunk_embeddings.npy", chunk_embeddings)
    np.save(OUTPUT_DIR / "paper_embeddings.npy", paper_embeddings)
    np.save(OUTPUT_DIR / "student_embeddings.npy", student_embeddings)

    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "chunk_size_words": CHUNK_SIZE_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
            "n_chunks": len(all_chunks),
            "n_papers": len(paper_meta),
            "n_students": len(student_meta),
            "chunk_meta": chunk_meta,
            "paper_meta": paper_meta,
            "student_meta": student_meta,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到: {OUTPUT_DIR}")
    print(f"  chunk 嵌入: {chunk_embeddings.shape}")
    print(f"  paper 嵌入: {paper_embeddings.shape}")
    print(f"  student 嵌入: {student_embeddings.shape}")

    # ---- 5.8 相似度分析 ----
    write_similarity_report(student_embeddings, student_meta, paper_embeddings, paper_meta)


def write_similarity_report(student_emb: np.ndarray,
                            student_meta: List[Dict],
                            paper_emb: np.ndarray,
                            paper_meta: List[Dict]):
    """生成相似度报告：学生间、论文间、以及同学生内部一致性。"""
    lines = []

    # 学生两两相似度矩阵
    sim = student_emb @ student_emb.T
    lines.append("=" * 60)
    lines.append("学生写作风格相似度矩阵（余弦相似度，1=最相似）")
    lines.append("=" * 60)

    ids = [m["student_id"] for m in student_meta]
    header = "            " + "  ".join(f"{sid:>8}" for sid in ids)
    lines.append(header)
    for i, sid in enumerate(ids):
        row = f"{sid:>10}  " + "  ".join(f"{sim[i, j]:8.3f}" for j in range(len(ids)))
        lines.append(row)

    # 每个学生内部论文一致性（多篇论文才有意义）
    lines.append("")
    lines.append("=" * 60)
    lines.append("学生内部论文风格一致性（同学生多篇论文平均两两相似度）")
    lines.append("=" * 60)
    for i, m in enumerate(student_meta):
        if m["n_papers"] < 2:
            lines.append(f"{m['student_id']}: 只有 1 篇论文，跳过")
            continue
        idxs = [j for j, pm in enumerate(paper_meta)
                if pm["student_id"] == m["student_id"]]
        sub = paper_emb[idxs] @ paper_emb[idxs].T
        n = len(idxs)
        avg = (sub.sum() - n) / (n * (n - 1))  # 去掉对角线
        lines.append(f"{m['student_id']}: {avg:.3f}  (n_papers={n})")

    # 论文两两最相似对
    lines.append("")
    lines.append("=" * 60)
    lines.append("最相似的论文对 Top 20")
    lines.append("=" * 60)
    psim = paper_emb @ paper_emb.T
    pairs = []
    for i in range(len(paper_meta)):
        for j in range(i + 1, len(paper_meta)):
            pairs.append((psim[i, j], paper_meta[i], paper_meta[j]))
    pairs.sort(key=lambda x: -x[0])
    for s, m1, m2 in pairs[:20]:
        lines.append(f"{s:.3f}  {m1['student_id']}/{m1['paper_file']}"
                     f"  <->  {m2['student_id']}/{m2['paper_file']}")

    report = "\n".join(lines)
    out = OUTPUT_DIR / "similarity_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"\n相似度报告已保存到: {out}")
    print("\n" + report[:2000])


if __name__ == "__main__":
    main()