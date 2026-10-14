# -*- coding: utf-8 -*-
"""
基于可解释写作特征聚类 + 命名 + 可视化
嵌入保留用于生成阶段，聚类改用特征空间

修改要点：
- 学生唯一化：优先使用 metadata.json 中的 student_meta，避免同一学生多篇论文重复参与聚类。
- 学生画像改为绑定到唯一学生，不再强行绑定单篇论文。
- 修正 student_embeddings 的索引映射，避免过滤学生后索引错位。
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============ 输出目录改到 essay_ethics_2023_2024_2025 下面 ============
OUT = Path(r"C:\Users\lcjessicazhu\OneDrive - HKUST\Desktop\essay_ethics_2023_2024_2025\style_analysis\outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

K = 8
RANDOM_SEED = 42

# 用于聚类的特征（全部 27 个，或精选子集）
CLUSTER_FEATURES = [
    "type_token_ratio", "avg_word_length", "long_word_ratio",
    "rare_word_ratio", "avg_sentence_length", "std_sentence_length",
    "long_sentence_ratio", "short_sentence_ratio", "clause_marker_density",
    "connective_density", "first_person_ratio", "second_person_ratio",
    "modal_density", "comma_freq", "semicolon_freq", "dash_freq",
    "parenthesis_freq", "question_freq", "exclamation_freq",
    "flesch_reading_ease", "flesch_kincaid_grade",
    "avg_paragraph_length", "sentences_per_paragraph",
]

FEATURE_LABELS = {
    "type_token_ratio":       ("Diverse-Vocab",       "Repetitive-Vocab"),
    "avg_word_length":        ("Formal-Lexis",        "Simple-Lexis"),
    "long_word_ratio":        ("Sophisticated-Lexis", "Basic-Lexis"),
    "rare_word_ratio":        ("Distinctive-Lexis",   "Common-Lexis"),
    "avg_sentence_length":    ("Long-Sentence",       "Short-Sentence"),
    "std_sentence_length":    ("Variable-Rhythm",     "Uniform-Rhythm"),
    "long_sentence_ratio":    ("Complex-Syntax",      "Simple-Syntax"),
    "short_sentence_ratio":   ("Punchy-Syntax",       "Flowing-Syntax"),
    "clause_marker_density":  ("Clause-Heavy",        "Linear-Syntax"),
    "connective_density":     ("Connective-Rich",     "Connective-Sparse"),
    "first_person_ratio":     ("Personal-Voice",      "Impersonal"),
    "second_person_ratio":    ("Direct-Address",      "No-Address"),
    "modal_density":          ("Hedged-Tone",         "Assertive-Tone"),
    "comma_freq":             ("Comma-Heavy",         "Comma-Light"),
    "semicolon_freq":         ("Semicolon-User",      "No-Semicolon"),
    "dash_freq":              ("Dash-User",           "No-Dash"),
    "parenthesis_freq":       ("Parenthetical",       "No-Parenthesis"),
    "question_freq":          ("Questioning",         "Declarative"),
    "exclamation_freq":       ("Emphatic",            "Restrained"),
    "flesch_reading_ease":    ("Accessible",          "Dense"),
    "flesch_kincaid_grade":   ("Advanced-Grade",      "Basic-Grade"),
    "avg_paragraph_length":   ("Long-Paragraphs",     "Short-Paragraphs"),
    "sentences_per_paragraph":("Multi-Sentence-Para", "Single-Sentence-Para"),
}

RADAR_FEATURES = [
    "type_token_ratio", "avg_word_length",
    "avg_sentence_length", "std_sentence_length",
    "clause_marker_density", "connective_density",
    "first_person_ratio", "modal_density",
]

# ============ 加载 ============
student_emb_path = OUT / "student_embeddings.npy"
paper_emb_path = OUT / "paper_embeddings.npy"

student_emb = np.load(student_emb_path) if student_emb_path.exists() else None
paper_emb = np.load(paper_emb_path) if paper_emb_path.exists() else None

with open(OUT / "metadata.json", encoding="utf-8") as f:
    meta = json.load(f)
with open(OUT / "student_features.json", encoding="utf-8") as f:
    sfeat = json.load(f)

pmeta = meta["paper_meta"]
student_meta = meta.get("student_meta", [])

# ---------- 学生唯一化 ----------
if student_meta:
    students = [s["student_id"] for s in student_meta]
else:
    # 回退：从 paper_meta 去重，保持首次出现顺序
    students = list(dict.fromkeys(p["student_id"] for p in pmeta))

# 只保留在 sfeat 中存在的学生（防止缺特征）
students = [s for s in students if s in sfeat]
n = len(students)
print(f"参与聚类的唯一学生数: {n}")

# ============ 特征矩阵 ============
X_raw = np.array([[sfeat[s].get(k, 0.0) for k in CLUSTER_FEATURES]
                  for s in students])

# 标准化（必须！不同特征量纲不同）
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
X_z = X  # 已经是 z-score

# 防止极端值影响（可选）
X = np.clip(X, -3, 3)

print(f"特征矩阵: {X.shape}")
print("各特征标准差（标准化后应接近 1）:")
for i, k in enumerate(CLUSTER_FEATURES):
    print(f"  {k:28s} std={X[:,i].std():.3f}  range=[{X[:,i].min():.2f}, {X[:,i].max():.2f}]")

# ============ 聚类 ============
km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=20)
labels = km.fit_predict(X)

# ============ 命名 ============
def auto_name(z_row, feature_names, top_n=2):
    idx = np.argsort(-np.abs(z_row))[:top_n]
    parts = []
    for i in idx:
        f = feature_names[i]
        z = z_row[i]
        if f in FEATURE_LABELS:
            high, low = FEATURE_LABELS[f]
            parts.append(high if z > 0 else low)
        else:
            parts.append(f"{f}+" if z > 0 else f"{f}-")
    return " & ".join(parts)

cluster_members = defaultdict(list)
for i, l in enumerate(labels):
    cluster_members[int(l)].append(i)

cluster_profiles = {}
for cid in range(K):
    idxs = cluster_members[cid]
    z_center = X_z[idxs].mean(axis=0)
    name = auto_name(z_center, CLUSTER_FEATURES, top_n=2)
    mean_raw = X_raw[idxs].mean(axis=0)
    cluster_profiles[cid] = {
        "cluster_id": cid,
        "auto_name": name,
        "n_students": len(idxs),
        "mean_features": {k: float(v) for k, v in zip(CLUSTER_FEATURES, mean_raw)},
        "z_scores": {k: float(v) for k, v in zip(CLUSTER_FEATURES, z_center)},
        "students": [students[i] for i in idxs],
    }

# 打印命名结果
print("\n" + "=" * 70)
print("Cluster 命名结果")
print("=" * 70)
for cid in range(K):
    p = cluster_profiles[cid]
    z = p["z_scores"]
    top_high = sorted(z.items(), key=lambda x: -x[1])[:3]
    top_low = sorted(z.items(), key=lambda x: x[1])[:3]
    print(f"\nC{cid}: {p['auto_name']}  (n={p['n_students']})")
    print(f"  High: " + ", ".join(f"{k}({v:+.2f})" for k, v in top_high))
    print(f"  Low:  " + ", ".join(f"{k}({v:+.2f})" for k, v in top_low))

with open(OUT / "cluster_profiles_feature_based.json", "w", encoding="utf-8") as f:
    json.dump(cluster_profiles, f, ensure_ascii=False, indent=2)

# ============ 可视化 ============
# 用 PCA 在特征空间降维
pca_feat = PCA(n_components=2, random_state=RANDOM_SEED)
coords = pca_feat.fit_transform(X)
print(f"\n特征空间 PCA 解释方差: {pca_feat.explained_variance_ratio_.sum():.1%}")

cmap = plt.get_cmap("tab10", K)
fig, ax = plt.subplots(figsize=(13, 10))
for cid in range(K):
    idxs = cluster_members[cid]
    pts = coords[idxs]
    ax.scatter(pts[:, 0], pts[:, 1], color=cmap(cid), s=50, alpha=0.75,
               edgecolors="white", linewidths=0.5,
               label=f"C{cid}: {cluster_profiles[cid]['auto_name']} (n={len(idxs)})")
    cen = pts.mean(axis=0)
    ax.annotate(f"C{cid}", (cen[0], cen[1]), fontsize=14, fontweight="bold",
                color=cmap(cid), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=cmap(cid), lw=2, alpha=0.95))
ax.set_title(f"Writing-Style Clusters (Feature Space PCA, K={K})", fontsize=15)
ax.set_xlabel("PC1 (style)"); ax.set_ylabel("PC2 (style)")
ax.legend(loc="best", fontsize=8, ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG / "30_clusters_feature_pca.png", dpi=150)
plt.close()
print(f"Saved: {FIG / '30_clusters_feature_pca.png'}")

# t-SNE 在特征空间
tsne = TSNE(n_components=2, perplexity=min(30, max(5, n - 1)),
            random_state=RANDOM_SEED, init="pca")
coords_tsne = tsne.fit_transform(X)

fig, ax = plt.subplots(figsize=(13, 10))
for cid in range(K):
    idxs = cluster_members[cid]
    pts = coords_tsne[idxs]
    ax.scatter(pts[:, 0], pts[:, 1], color=cmap(cid), s=50, alpha=0.75,
               edgecolors="white", linewidths=0.5,
               label=f"C{cid}: {cluster_profiles[cid]['auto_name']} (n={len(idxs)})")
    cen = pts.mean(axis=0)
    ax.annotate(f"C{cid}", (cen[0], cen[1]), fontsize=14, fontweight="bold",
                color=cmap(cid), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=cmap(cid), lw=2, alpha=0.95))
ax.set_title(f"Writing-Style Clusters (Feature Space t-SNE, K={K})", fontsize=15)
ax.legend(loc="best", fontsize=8, ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(FIG / "31_clusters_feature_tsne.png", dpi=150)
plt.close()
print(f"Saved: {FIG / '31_clusters_feature_tsne.png'}")

# ============ 雷达图 ============
vals_all = np.array([[sfeat[s].get(k, 0) for k in RADAR_FEATURES] for s in students])
vmin, vmax = vals_all.min(0), vals_all.max(0)
norm_all = (vals_all - vmin) / (vmax - vmin + 1e-9)

n_cols = 4
n_rows = (K + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows),
                         subplot_kw=dict(polar=True))
axes = axes.flatten()
angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATURES), endpoint=False).tolist()
angles += angles[:1]
for cid in range(K):
    ax = axes[cid]
    idxs = cluster_members[cid]
    mean_norm = norm_all[idxs].mean(axis=0)
    row = mean_norm.tolist() + [mean_norm[0]]
    ax.plot(angles, row, color=cmap(cid), linewidth=2)
    ax.fill(angles, row, color=cmap(cid), alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n") for f in RADAR_FEATURES], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"C{cid}: {cluster_profiles[cid]['auto_name']}\n(n={len(idxs)})",
                 fontsize=9, fontweight="bold")
for j in range(K, len(axes)):
    axes[j].axis("off")
plt.suptitle("Representative Writing Style per Cluster (Feature-Space)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "32_radar_feature_clusters.png", dpi=150)
plt.close()
print(f"Saved: {FIG / '32_radar_feature_clusters.png'}")

# ============ 热力图 ============
cluster_z = np.array([X_z[cluster_members[cid]].mean(axis=0) for cid in range(K)])
fig, ax = plt.subplots(figsize=(14, max(4, 0.55 * K + 1)))
im = ax.imshow(cluster_z, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
ax.set_xticks(range(len(CLUSTER_FEATURES)))
ax.set_xticklabels(CLUSTER_FEATURES, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(K))
ax.set_yticklabels([f"C{cid}: {cluster_profiles[cid]['auto_name']}"
                    for cid in range(K)], fontsize=8)
for i in range(K):
    for j in range(len(CLUSTER_FEATURES)):
        v = cluster_z[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                fontsize=6, color="white" if abs(v) > 1.3 else "black")
plt.colorbar(im, ax=ax, label="z-score (vs. all students)")
ax.set_title("Cluster Feature Profile (z-scores, feature-space clustering)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "33_feature_heatmap_v2.png", dpi=150)
plt.close()
print(f"Saved: {FIG / '33_feature_heatmap_v2.png'}")

# ============ 更新学生档案 ============
# 学生级画像，绑定唯一学生；嵌入使用 student_embeddings（若存在）
student_meta_map = {s["student_id"]: s for s in student_meta} if student_meta else {}

# 修正：student_id -> student_embeddings 中的索引
sid_to_emb_idx = {s["student_id"]: idx for idx, s in enumerate(student_meta)} if student_meta else {}

profiles = {}
for i, sid in enumerate(students):
    cid = int(labels[i])
    prof = {
        "student_id": sid,
        "cluster_id": cid,
        "cluster_name": cluster_profiles[cid]["auto_name"],
        "style_features": sfeat.get(sid, {}),
        "paper_files": student_meta_map.get(sid, {}).get("paper_files",
                                                         sfeat.get(sid, {}).get("paper_files", [])),
    }
    if student_emb is not None and sid in sid_to_emb_idx:
        emb_idx = sid_to_emb_idx[sid]
        if emb_idx < len(student_emb):
            prof["style_embedding"] = student_emb[emb_idx].tolist()
    profiles[sid] = prof

with open(OUT / "student_style_profiles_v3.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {OUT / 'student_style_profiles_v3.json'}")
print("Done.")