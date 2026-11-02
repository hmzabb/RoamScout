"""切分策略 A/B 对比评估：召回率 / 正确率 / 延迟。

三种策略 × 不同参数 → 18 组实验，基于 QA_PAIRS 评测集统一评分。

用法（changan_ai 目录下）：
    .venv/python.exe scripts/eval_chunking_ab.py

输出：
    scripts/eval_chunking_ab_<timestamp>.txt（人类可读报告）
    scripts/eval_chunking_ab_<timestamp>.json（结构化数据，方便画图）
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, BASE_DIR
from app.repositories.vector_store import get_store
from app.services.chunking import (
    Chunk,
    ParentChunk,
    chunk_markdown,
    chunk_plain_text,
    chunk_recursive,
    chunk_parent_child,
)
from app.services.embedding import get_embedding_client
from app.services.quality.readers import extract_text, supported_extensions
from scripts.qa_pairs import QA_PAIRS, QAPair

# ---- 输出 ----
TS = int(time.time())
OUTPUT_TXT = Path(__file__).with_name(f"eval_chunking_ab_{TS}.txt")
OUTPUT_JSON = Path(__file__).with_name(f"eval_chunking_ab_{TS}.json")
CORPUS_DIR = BASE_DIR / "app" / "data" / "corpus"
EMBED_BATCH = 16
RECALL_K = 4
_PAD = "=" * 72


def log(msg: str) -> None:
    print(msg)
    with open(OUTPUT_TXT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


OUTPUT_TXT.write_text("", encoding="utf-8")


# ============================================================
# 实验配置（18 组）
# ============================================================
@dataclass
class ChunkConfig:
    name: str
    family: str          # "recursive" / "semantic" / "parent_child"
    chunk_size: int = 0   # Child size for parent-child, main chunk size otherwise
    overlap: int = 0
    parent_strategy: str = ""  # "doc" / "fixed" / "section"


CONFIGS: list[ChunkConfig] = [
    # ── 递归切分（6 组）──
    ChunkConfig("recursive-200-o40",    "recursive",  200, 40),
    ChunkConfig("recursive-300-o60",    "recursive",  300, 60),
    ChunkConfig("recursive-500-o100",   "recursive",  500, 100),
    ChunkConfig("recursive-500-o50",    "recursive",  500, 50),
    ChunkConfig("recursive-800-o160",   "recursive",  800, 160),
    ChunkConfig("recursive-1000-o0",    "recursive", 1000, 0),

    # ── 结构化感知切分（6 组）──
    ChunkConfig("semantic-300-o60",     "semantic",   300, 60),
    ChunkConfig("semantic-400-o80",     "semantic",   400, 80),
    ChunkConfig("semantic-600-o120",    "semantic",   600, 120),  # BASELINE
    ChunkConfig("semantic-600-o60",     "semantic",   600, 60),
    ChunkConfig("semantic-800-o160",    "semantic",   800, 160),
    ChunkConfig("semantic-1000-o200",   "semantic",  1000, 200),

    # ── 父子文档（6 组）──
    ChunkConfig("pc-200-doc",           "parent_child", 200, 40,  "doc"),
    ChunkConfig("pc-300-doc",           "parent_child", 300, 60,  "doc"),
    ChunkConfig("pc-400-doc",           "parent_child", 400, 80,  "doc"),
    ChunkConfig("pc-600-doc",           "parent_child", 600, 120, "doc"),
    ChunkConfig("pc-400-1200",          "parent_child", 400, 80,  "fixed"),
    ChunkConfig("pc-400-section",       "parent_child", 400, 80,  "section"),
]


# ============================================================
# 数据加载：读取 corpus 目录下所有文件
# ============================================================
def load_corpus_texts() -> list[tuple[str, str, str]]:
    """返回 [(text, doc_title, file_name), ...]"""
    docs: list[tuple[str, str, str]] = []
    for ext in supported_extensions():
        for file_path in sorted(CORPUS_DIR.rglob(f"*{ext}")):
            try:
                text = extract_text(file_path)
            except Exception:
                continue
            if not text.strip():
                continue
            docs.append((text, file_path.stem, file_path.name))
    return docs


# ============================================================
# 切分：根据配置选策略
# ============================================================
def chunk_docs(
    docs: list[tuple[str, str, str]],
    config: ChunkConfig,
) -> tuple[list[Chunk], dict[str, ParentChunk]]:
    """按配置对所有文档执行切分，返回 (child_chunks, parent_map)。

    非 parent_child 策略返回空 parent_map。
    """
    all_chunks: list[Chunk] = []
    parent_map: dict[str, ParentChunk] = {}

    for text, doc_title, file_name in docs:
        if config.family == "recursive":
            chunks = chunk_recursive(
                text, doc_title, file_name,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
            )
            all_chunks.extend(chunks)

        elif config.family == "semantic":
            ext = Path(file_name).suffix.lower()
            if ext == ".md":
                # 临时覆盖 CHUNK_MAX / OVERLAP 来做参数实验
                import app.services.chunking as cmod
                saved_max, saved_overlap = cmod.CHUNK_MAX, cmod.OVERLAP
                cmod.CHUNK_MAX = config.chunk_size
                cmod.OVERLAP = config.overlap
                try:
                    chunks = chunk_markdown(text, doc_title, file_name)
                finally:
                    cmod.CHUNK_MAX, cmod.OVERLAP = saved_max, saved_overlap
            else:
                import app.services.chunking as cmod
                saved_max, saved_overlap = cmod.CHUNK_MAX, cmod.OVERLAP
                cmod.CHUNK_MAX = config.chunk_size
                cmod.OVERLAP = config.overlap
                try:
                    chunks = chunk_plain_text(text, doc_title, file_name)
                finally:
                    cmod.CHUNK_MAX, cmod.OVERLAP = saved_max, saved_overlap
            all_chunks.extend(chunks)

        elif config.family == "parent_child":
            children, parents = chunk_parent_child(
                text, doc_title, file_name,
                child_size=config.chunk_size,
                parent_strategy=config.parent_strategy,
                overlap=config.overlap,
            )
            all_chunks.extend(children)
            parent_map.update(parents)

    return all_chunks, parent_map


# ============================================================
# 入库 + 评估
# ============================================================
@dataclass
class EvalResult:
    config_name: str
    family: str
    num_chunks: int
    num_parents: int
    recall: float
    precision_at_k: float
    avg_latency_ms: float
    valid_queries: int
    layer_breakdown: dict[int, dict] = field(default_factory=dict)


def run_eval(
    config: ChunkConfig,
    docs: list[tuple[str, str, str]],
    qa_pairs: list[QAPair],
    store,
    embedder,
) -> EvalResult:
    """对一个配置完成 切分 → 入库 → 评估 全流程。"""
    log(f"\n{'─'*60}")
    log(f"  [{config.name}] 切分中...")

    # 1. 切分
    t0 = time.time()
    all_chunks, parent_map = chunk_docs(docs, config)
    chunk_time = (time.time() - t0) * 1000
    log(f"  [{config.name}] 切分完成: {len(all_chunks)} Child chunks, "
        f"{len(parent_map)} Parents, 耗时 {chunk_time:.0f}ms")

    # 2. 清空 + embed + 入库
    store.delete_by_source("corpus")
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i: i + EMBED_BATCH]
        embeddings = embedder.embed_texts([c.text for c in batch])
        store.upsert_chunks(batch, embeddings)

    log(f"  [{config.name}] 入库完成: {len(all_chunks)} chunks")

    # 3. 评估
    log(f"  [{config.name}] 评估 {len(qa_pairs)} 个 QA...")
    is_parent_child = config.family == "parent_child"

    recall_hits = 0
    precision_sum = 0.0
    total_latency = 0.0
    valid_queries = 0
    layer_stats: dict[int, dict] = defaultdict(lambda: {"hits": 0, "total": 0, "precision_sum": 0.0, "latency_sum": 0.0})

    for qa in qa_pairs:
        if qa.is_boundary:
            continue  # 边界 case 不算在核心指标里

        expected_docs = set(qa.relevant_docs)
        t_start = time.perf_counter()

        qv = embedder.embed_texts([qa.query])[0]
        hits = store.query(qv, top_k=RECALL_K, where={"source": "corpus"})
        hits = [dict(h, similarity=h["score"], milvus_sim=h["score"]) for h in hits]
        hits = [h for h in hits if h["milvus_sim"] >= settings.rag_recall_min_score]

        if is_parent_child:
            # 父子文档：命中 Child → 查 parent_id → 返回 Parent
            seen_parents: set[str] = set()
            resolved_hits: list[dict] = []
            for h in hits:
                pid = h["metadata"].get("parent_id", "")
                if pid and pid in parent_map and pid not in seen_parents:
                    seen_parents.add(pid)
                    parent = parent_map[pid]
                    resolved_hits.append({
                        "text": parent.text,
                        "metadata": parent.metadata,
                        "similarity": h["similarity"],
                    })
                if len(resolved_hits) >= RECALL_K:
                    break
            hits = resolved_hits

        elapsed = (time.perf_counter() - t_start) * 1000
        total_latency += elapsed
        layer_stats[qa.layer]["latency_sum"] += elapsed

        retrieved_docs = [h.get("metadata", {}).get("doc", "") for h in hits]
        hit = len(expected_docs & set(retrieved_docs)) > 0
        if hit:
            recall_hits += 1

        relevant_in_k = sum(1 for d in retrieved_docs if d in expected_docs)
        precision = relevant_in_k / len(hits) if hits else 0.0
        precision_sum += precision

        valid_queries += 1
        layer_stats[qa.layer]["hits"] += 1 if hit else 0
        layer_stats[qa.layer]["total"] += 1
        layer_stats[qa.layer]["precision_sum"] += precision

    recall = recall_hits / valid_queries if valid_queries else 0.0
    precision_at_k = precision_sum / valid_queries if valid_queries else 0.0
    avg_latency = total_latency / valid_queries if valid_queries else 0.0

    result = EvalResult(
        config_name=config.name,
        family=config.family,
        num_chunks=len(all_chunks),
        num_parents=len(parent_map),
        recall=recall,
        precision_at_k=precision_at_k,
        avg_latency_ms=avg_latency,
        valid_queries=valid_queries,
        layer_breakdown={
            layer: {
                "recall": s["hits"] / s["total"] if s["total"] else 0.0,
                "precision": s["precision_sum"] / s["total"] if s["total"] else 0.0,
                "avg_latency_ms": s["latency_sum"] / s["total"] if s["total"] else 0.0,
                "count": s["total"],
            }
            for layer, s in sorted(layer_stats.items())
        },
    )
    return result


# ============================================================
# 报告输出
# ============================================================
def print_report(results: list[EvalResult]) -> None:
    log(f"\n{_PAD}")
    log("  切分策略 A/B 对比评估报告")
    log(f"{_PAD}")
    log(f"  评测集: {results[0].valid_queries} 个 QA（去边界）")
    log(f"  召回 Top-K: {RECALL_K}")
    log(f"  Corpus 文件: {len(list(CORPUS_DIR.rglob('*.md'))) + len(list(CORPUS_DIR.rglob('*.pdf'))) + len(list(CORPUS_DIR.rglob('*.docx')))}")
    log(f"  模型: {settings.embedding_model}")
    log("")

    # 总览表
    header = f"  {'策略':<26s} {'Chunks':>7s} {'Parents':>8s} {'Recall':>8s} {'Prec@K':>8s} {'延迟(ms)':>9s}"
    log(header)
    log(f"  {'─'*70}")

    sorted_results = sorted(results, key=lambda r: r.recall, reverse=True)

    for r in sorted_results:
        marker = " ⭐" if r.config_name == "semantic-600-o120" else "  "
        log(
            f"{marker} {r.config_name:<24s} "
            f"{r.num_chunks:>7d} "
            f"{r.num_parents:>8d} "
            f"{r.recall:>7.1%} "
            f"{r.precision_at_k:>7.1%} "
            f"{r.avg_latency_ms:>8.0f}"
        )

    log("")

    # 按家族汇总
    log(f"  {'─'*70}")
    log("  按策略家族汇总:")
    log(f"  {'家族':<16s} {'平均Recall':>11s} {'平均Prec':>10s} {'平均延迟':>9s}")
    log(f"  {'─'*50}")
    for family in ["recursive", "semantic", "parent_child"]:
        family_results = [r for r in results if r.family == family]
        if not family_results:
            continue
        avg_r = sum(r.recall for r in family_results) / len(family_results)
        avg_p = sum(r.precision_at_k for r in family_results) / len(family_results)
        avg_l = sum(r.avg_latency_ms for r in family_results) / len(family_results)
        log(f"  {family:<16s} {avg_r:>10.1%} {avg_p:>9.1%} {avg_l:>8.0f}ms")

    log("")

    # 分层明细（最优 vs BASELINE）
    baseline = next((r for r in results if r.config_name == "semantic-600-o120"), None)
    best = sorted_results[0]
    log(f"  {'─'*70}")
    log("  分层对比（BASELINE vs 最优）:")
    for layer in [1, 2, 3, 4]:
        bl = baseline.layer_breakdown.get(layer, {})
        bt = best.layer_breakdown.get(layer, {})
        if not bl or not bt:
            continue
        log(f"  Layer {layer} ({bl.get('count', 0)} QA):")
        log(f"    BASELINE: Recall={bl['recall']:.1%}  Prec={bl['precision']:.1%}  Lat={bl['avg_latency_ms']:.0f}ms")
        log(f"    最优:     Recall={bt['recall']:.1%}  Prec={bt['precision']:.1%}  Lat={bt['avg_latency_ms']:.0f}ms")

    log("")

    # Top-3 推荐
    log(f"  {'─'*70}")
    log("  Top-3 推荐:")
    for i, r in enumerate(sorted_results[:3]):
        log(f"  #{i+1} {r.config_name} — Recall={r.recall:.1%} Prec@{RECALL_K}={r.precision_at_k:.1%} Lat={r.avg_latency_ms:.0f}ms")


# ============================================================
# Main
# ============================================================
def main() -> None:
    log(f"{_PAD}")
    log("  切分策略 A/B 对比评估")
    log(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  输出文件: {OUTPUT_TXT.name}")
    log(f"{_PAD}")

    # 加载数据
    log("\n[1/3] 加载 corpus 文本...")
    docs = load_corpus_texts()
    log(f"  共 {len(docs)} 个文档")

    # 加载 QA
    qa_pairs = [qa for qa in QA_PAIRS if not qa.is_boundary]
    boundary_pairs = [qa for qa in QA_PAIRS if qa.is_boundary]
    log(f"  评测集: {len(qa_pairs)} 有效 QA + {len(boundary_pairs)} 边界 case")

    # 初始化
    store = get_store()
    embedder = get_embedding_client()

    # 跑实验
    log(f"\n[2/3] 开始 {len(CONFIGS)} 组实验...")
    results: list[EvalResult] = []
    for idx, config in enumerate(CONFIGS):
        log(f"\n  ▶ [{idx+1}/{len(CONFIGS)}] {config.name}")
        try:
            result = run_eval(config, docs, qa_pairs, store, embedder)
            results.append(result)
        except Exception as e:
            log(f"  ✗ {config.name} 失败: {e}")

    # 输出报告
    log(f"\n[3/3] 生成报告...")
    print_report(results)

    # JSON 输出
    json_output = {
        "timestamp": TS,
        "config": {
            "recall_k": RECALL_K,
            "embedding_model": settings.embedding_model,
            "num_queries": results[0].valid_queries if results else 0,
            "num_docs": len(docs),
        },
        "results": [
            {
                "name": r.config_name,
                "family": r.family,
                "num_chunks": r.num_chunks,
                "num_parents": r.num_parents,
                "recall": round(r.recall, 4),
                "precision_at_k": round(r.precision_at_k, 4),
                "avg_latency_ms": round(r.avg_latency_ms, 2),
                "layer_breakdown": {
                    str(l): {
                        "recall": round(d["recall"], 4),
                        "precision": round(d["precision"], 4),
                        "avg_latency_ms": round(d["avg_latency_ms"], 2),
                        "count": d["count"],
                    }
                    for l, d in r.layer_breakdown.items()
                },
            }
            for r in results
        ],
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    log(f"\n  报告已输出:")
    log(f"    {OUTPUT_TXT}")
    log(f"    {OUTPUT_JSON}")
    log(f"\n{_PAD}")


if __name__ == "__main__":
    main()