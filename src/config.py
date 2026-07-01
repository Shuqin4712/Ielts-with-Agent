"""集中式配置：模型名、base_url、路径、embedding 后端等常量。

设计原则：所有「会变的旋钮」都收在这里，业务代码只 import 常量，
不在各处散落 magic string。密钥只从环境变量读，绝不写进代码。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 在 import 期就加载 .env，让后续 os.getenv 能读到。
# override=True：让 .env 成为本项目密钥的唯一权威，覆盖可能残留的、
# 过期的系统级环境变量（否则改了 .env 却被旧系统变量盖掉，极难排查）。
load_dotenv(override=True)

# ── 路径 ────────────────────────────────────────────────────────────
# PROJECT_ROOT = 仓库根（本文件在 src/ 下，往上一级）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 持久化位置：ChromaDB 向量库 + SQLite 主库。放 data/ 下、且已被 .gitignore 忽略。
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "ielts.sqlite"
GOLD_DIR = DATA_DIR / "gold"            # 剑桥 gold：manifest.json + bodies/<id>.txt（人工转写）

# ── DeepSeek（OpenAI 兼容）─────────────────────────────────────────
# 经 langchain-openai 的 ChatOpenAI 接入：只需改 base_url + 模型名。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 成本路由：简单工具用 flash，判分/反思用 pro(+thinking)。
# 注意：不要用旧名 deepseek-chat / deepseek-reasoner（2026-07-24 停用）。
MODEL_FLASH = "deepseek-v4-flash"   # 快/省，1M 上下文
MODEL_PRO = "deepseek-v4-pro"       # 推理/agentic

# ── 本地 Embedding（DeepSeek 不提供 embedding）─────────────────────
# 后端目前只支持 "ollama"。模型可在 bge-m3 / nomic-embed-text 间切换：
#   bge-m3          中英双语强，本项目首选
#   nomic-embed-text 备选（已就绪，想省下载时用）
EMBED_BACKEND = "ollama"
EMBED_MODEL = "bge-m3"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── ChromaDB 集合名 ────────────────────────────────────────────────
COLL_RUBRIC = "rubric_descriptors"     # metadata: criterion, band, task_type
COLL_EXEMPLAR = "exemplar_essays"      # metadata: task_type, band, topic, tier

# ── 阶段 0 数据切分 ────────────────────────────────────────────────
SPLIT_SEED = 42          # 切分可复现
HOLDOUT_FRAC = 0.12      # 从 silver 留作临时评测基线的比例（分层抽样）
EXEMPLAR_PER_BUCKET = 2  # 每个 (task_type × band) 桶选几篇当锚点，控制锚点总量


def require_api_key() -> str:
    """需要 DeepSeek 时调用，缺失就明确报错而非静默失败。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入你的 key。"
        )
    return DEEPSEEK_API_KEY
