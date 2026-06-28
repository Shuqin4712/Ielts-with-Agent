# CLAUDE.md — IELTS Writing Agent 项目共识

> 这是给 Claude Code 的项目上下文。深度设计见 `DESIGN.md`。
> 本文件钉死了已做的决策，**不要在编码中推翻它们**；如需改动先和我确认。

## 一句话

一个基于 LangGraph 的雅思写作（Task 1 + 2）批改与学习 Agent：批改模式按官方四维（TA/CC/LR/GRA）打分并给依据，助手模式按需调用工具，配套量化评测与跨会话记忆。定位是**可落地、有真实价值**的全栈项目。

## 技术栈（钉死）

- 编排：**LangGraph**（StateGraph + 子图 + 条件边 + checkpointer）
- LLM：**DeepSeek API**，OpenAI 兼容，`base_url="https://api.deepseek.com"`，经 `langchain-openai` 的 `ChatOpenAI` 接入
- Embedding：**本地** `bge-m3`（或 Ollama 的 `nomic-embed-text`）——DeepSeek 不提供 embedding
- 向量库：**ChromaDB** ｜ 结构化存储：**SQLite**
- 后端：**FastAPI**（`/chat` 走 SSE 流式）｜ 前端：**纯 HTML/CSS/JS**
- 包管理：用 `uv` 或 venv；密钥走 `.env`（`DEEPSEEK_API_KEY`），**绝不硬编码进代码**

### DeepSeek 注意事项（容易踩坑）

- 模型名用 `deepseek-v4-flash`（快/省，1M 上下文）与 `deepseek-v4-pro`（推理/agentic）。
  **不要**用旧名 `deepseek-chat` / `deepseek-reasoner`（2026-07-24 停用）。
- thinking 模式：`reasoning_effort` + `extra_body={"thinking": {"type": "enabled"}}`；该模式下 `temperature` 等采样参数无效，不要设。
- 支持 OpenAI 风格 function calling（`tools` 参数）。
- 有并发限流（HTTP 429），调用要带指数退避重试。

## 架构（两模式，共用底层）

- **批改模式 = 确定性 workflow**：四维每次必跑，不让 LLM 自己决定跳过。流程：`ingest → planner → retrieve_rubric/exemplars → 四维打分 → aggregate → reflection(条件回环) → feedback+revision → memory_write`。
- **助手模式 = agentic**：LLM 经 function calling 自主选工具。
- 两模式共用 DeepSeek 模型、ChromaDB、SQLite。

## 关键设计决策（不要推翻）

1. **批改是 workflow，不是自主 agent**；只有 reflection 回环和助手模式的选 tool 是 agentic 的。
2. **Chunk 结构感知**：rubric 按 `(criterion, band)` 切并挂 metadata；范文按整篇切，metadata `{task_type, band, topic, tier}`。检索先 metadata 过滤再向量召回。
3. **打分用范文 in-context 锚定**（喂已知 band 的范文校准），不要让模型裸打分。
4. **评测只用 gold tier**（剑桥考官标注）当 ground truth；网络数据是 silver，不作基准。
5. **成本路由**：简单工具用 `v4-flash`，判分/反思用 `v4-pro` thinking。
6. `exemplar_provide`（生成范文）与 `deconstruct_article`（拆解用户文章）是两个不同工具，后者用 structured output 约束 JSON 抽取。
7. 记忆分短期（checkpointer）与长期（SQLite 学生画像：episodic + semantic）。

## 建议目录结构

```
ielts-writing-agent/
├── CLAUDE.md  DESIGN.md  pyproject.toml  .env.example
├── data/        raw/ processed/
├── src/
│   ├── config.py        # 模型名、路由、常量
│   ├── llm.py           # DeepSeek 封装 + 成本路由
│   ├── data/            # 归一化、入库
│   ├── rag/             # chunk / embed / chroma 检索
│   ├── graph/           # state / nodes / subgraphs / build
│   ├── tools/           # 工具集
│   ├── memory/          # 短期 + 长期
│   ├── eval/            # 评测 harness
│   ├── db/              # SQLite schema + CRUD
│   └── api/             # FastAPI
├── frontend/            # 纯 HTML/CSS/JS
└── tests/
```

## 开发阶段（先打通竖切，再加深）

- **[阶段 0] 数据地基** ← **当前在这里**：异构数据归一化进 SQLite；rubric/范文灌 ChromaDB；切 held-out 测试集。
- [阶段 1] 最薄竖切：最简 LangGraph 跑通四维打分出分（无 reflection/memory/锚定）。
- [阶段 2] 把打分做准：锚定 + reflection + eval harness（±0.5 一致率 / QWK + 锚定消融）。
- [阶段 3] 工具 + 助手模式 + 成本路由。
- [阶段 4] 记忆与个性化。
- [阶段 5] 前端 + 词库/素材库。
- [阶段 6] 可观测性 + README + 部署。

> 每个阶段产出一个能跑的东西再进下一阶段。改动 scope 或决策前先和我确认。

## 明确不做

口语/听力/阅读、多智能体架构、模型微调、用户系统/鉴权、移动端原生 App。

## 协作偏好

- 
- 写代码时请简要解释**为什么这么做**和**框架的使用模式**，而非堆砌底层细节。
- 沟通用中文，技术术语保留英文。
