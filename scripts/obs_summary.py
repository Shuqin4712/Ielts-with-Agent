"""阶段 6 可观测性：汇总 data/logs/llm_calls.jsonl → 用量 / 延迟 / 成本表。

读被动 callback 落下的每条 LLM 调用记录，按【操作】和【模型档位】两个维度汇总：
调用次数、token 用量、平均延迟、估算成本（用 config.PRICE_PER_MTOK）。
用来讲「成本路由（flash vs pro）省了多少、每类操作花在哪」。

用法：
  python scripts/obs_summary.py                # 全部
  python scripts/obs_summary.py --since 2026-07-02   # 只看某日期起（ISO 前缀匹配）
成本为估算值（价格是 config 里的占位常量，按官网改）。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.obs.tracker import LOG_PATH


def _cost(tier: str, pin: int, pout: int) -> float:
    p = config.PRICE_PER_MTOK.get(tier, {"input": 0, "output": 0})
    return (pin * p["input"] + pout * p["output"]) / 1_000_000


def load(since: str | None) -> list[dict]:
    if not LOG_PATH.exists():
        sys.exit(f"没有日志：{LOG_PATH}（先跑几次 grade/chat 生成记录）。")
    rows = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since and (r.get("ts") or "") < since:
            continue
        rows.append(r)
    return rows


def _agg(rows: list[dict], key: str) -> dict:
    """按某字段聚合：次数 / in / out / 总延迟 / 有延迟计数 / 成本 / 错误数。"""
    buckets: dict = defaultdict(lambda: {"n": 0, "in": 0, "out": 0, "lat": 0.0,
                                         "lat_n": 0, "cost": 0.0, "err": 0})
    for r in rows:
        b = buckets[r.get(key) or "—"]
        b["n"] += 1
        b["in"] += r.get("input_tokens") or 0
        b["out"] += r.get("output_tokens") or 0
        if r.get("latency_ms") is not None:
            b["lat"] += r["latency_ms"]
            b["lat_n"] += 1
        b["cost"] += _cost(r.get("tier") or "flash", r.get("input_tokens") or 0,
                           r.get("output_tokens") or 0)
        if r.get("error"):
            b["err"] += 1
    return buckets


def _print_table(title: str, buckets: dict, label: str) -> None:
    print(f"\n{title}")
    print(f"{label:<18}{'calls':>7}{'in_tok':>10}{'out_tok':>10}"
          f"{'avg_ms':>9}{'cost$':>10}{'err':>5}")
    print("-" * 69)
    for name in sorted(buckets, key=lambda k: -buckets[k]["cost"]):
        b = buckets[name]
        avg = b["lat"] / b["lat_n"] if b["lat_n"] else 0
        print(f"{name:<18}{b['n']:>7}{b['in']:>10}{b['out']:>10}"
              f"{avg:>9.0f}{b['cost']:>10.4f}{b['err']:>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="只统计该 ISO 时间前缀起的记录，如 2026-07-02")
    args = ap.parse_args()

    rows = load(args.since)
    if not rows:
        sys.exit("过滤后没有记录。")

    total_cost = sum(_cost(r.get("tier") or "flash", r.get("input_tokens") or 0,
                          r.get("output_tokens") or 0) for r in rows)
    total_in = sum(r.get("input_tokens") or 0 for r in rows)
    total_out = sum(r.get("output_tokens") or 0 for r in rows)

    print(f"日志：{LOG_PATH}")
    print(f"共 {len(rows)} 次调用 | 输入 {total_in:,} tok | 输出 {total_out:,} tok | "
          f"估算成本 ${total_cost:.4f}")
    print("（成本为估算，价格见 config.PRICE_PER_MTOK，请按 DeepSeek 官网校准）")

    _print_table("按操作（operation）：", _agg(rows, "op"), "operation")
    _print_table("按模型档位（tier）：", _agg(rows, "tier"), "tier")


if __name__ == "__main__":
    main()
