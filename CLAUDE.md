# CLAUDE.md — IELTS Writing Agent 项目共识

> 这是给 Claude Code 的项目上下文。深度设计见 `DESIGN.md`。
> 本文件钉死了已做的决策，**不要在编码中推翻它们**；如需改动先和我确认。

## 一句话

一个基于 LangGraph 的雅思写作（Task 1 + 2）批改与学习 Agent：批改模式按官方四维（TA/CC/LR/GRA）打分并给依据，助手模式按需调用工具，配套量化评测与跨会话记忆。定位是**可落地、有真实价值**的全栈项目。

## 技术栈

- 编排：**LangGraph**（StateGraph + 子图 + 条件边 + checkpointer）
- LLM：**DeepSeek API**，OpenAI 兼容，`base_url="https://api.deepseek.com"`，经 `langchain-openai` 的 `ChatOpenAI` 接入
- Embedding：**本地** `bge-m3`（Ollama）——DeepSeek 不提供 embedding
- 向量库：**ChromaDB** ｜ 结构化存储：**SQLite**
- 后端：**FastAPI**（`/chat` 走 SSE 流式）｜ 前端：**纯 HTML/CSS/JS**
- 包管理：用 `uv` 或 venv；密钥走 `.env`（`DEEPSEEK_API_KEY`），**绝不硬编码进代码**

### DeepSeek 注意事项

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

## 常用命令（已跑通）

环境：Python 3.13 + venv（非 uv）；本地 embedding 用 Ollama 的 `bge-m3`（需先 `ollama pull bge-m3`）。

```powershell
# 装依赖（首次）
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# 连通性 smoke test（DeepSeek / embedding / ChromaDB / SQLite 各一条）
python tests/test_connectivity.py        # 或 pytest tests/test_connectivity.py -v

# 阶段 0：构建数据地基（入 SQLite + 灌两个 ChromaDB 集合；不需 DeepSeek key）
python scripts/build_stage0.py
python scripts/demo_retrieve.py          # 验收：检索环境话题 band7 Task2 范文
pytest tests/test_stage0.py -v

# 阶段 1：最简 LangGraph 四维打分（需 DeepSeek key，会调 flash）
python scripts/grade_essay.py --gold --task 2   # 验收：贴一篇 gold 作文出四维+overall
pytest tests/test_stage1.py -v                  # LLM-free；全图 smoke 需 RUN_LLM_TESTS=1

# 阶段 2：eval harness（gold holdout 上量化；temp=0 可复现；并发+超时防卡死）
python -m src.eval.harness --config all         # baseline/anchored/reflect 全跑，写 results.jsonl
python -m src.eval.harness --compare            # 看历史对比表
pytest tests/test_stage2.py -v                  # 指标 + 泄漏断言（LLM-free）
# 检索层评测（v1.3）：范文检索 Recall@5/MRR@10 三档对比（只用本地 embedding，不调 DeepSeek）
python -m src.eval.retrieval                    # 查询集落 data/eval/retrieval_queries.jsonl，可人工校对
pytest tests/test_retrieval_eval.py -v          # 指标纯函数 + 查询集契约（LLM-free）

# 阶段 3：agentic 助手 REPL（LLM 自主选 tool，需 DeepSeek key）
python scripts/assistant.py                     # 对话：升级词 / 拆解文章 / 查词 / 打分 / 存库
pytest tests/test_stage3.py -v                  # 路由/CRUD LLM-free；工具+路由 smoke 需 RUN_LLM_TESTS=1

# 阶段 4：记忆与个性化（需 DeepSeek key）
python scripts/assistant.py --thread demo1      # 短期记忆：同 thread_id 多轮接续、落盘可续跑；--no-memory 关
python scripts/grade_essay.py --essay-id 123 --user alice   # 个性化全图：出反馈 + 更新画像；无 --user = 纯打分（与 eval 对齐）
python scripts/demo_memory.py                   # 演示：同 user 连批两篇，第二篇反馈「记得」第一篇（跑完清理）
pytest tests/test_stage4.py -v                  # 画像/checkpointer/护栏 LLM-free；反馈+蒸馏 smoke 需 RUN_LLM_TESTS=1
# 关键回归：关个性化跑纯打分管道，确认 QWK/±0.5 不漂移（记忆没污染打分）
python -m src.eval.harness --config anchored_flash --no-fourdim

# 阶段 5：起 web app（后端包住 LangGraph；前端由后端同源提供，无需单独起）
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
#   浏览器开 http://127.0.0.1:8000  → 五视图 web app（批改/对话/查词/词库/素材库）
#   http://127.0.0.1:8000/docs      → FastAPI 自带 Swagger，逐端点手测
# 开发时热重载（改 py 自动重启）：加 --reload
```

- 短期记忆落 `data/checkpoints.sqlite`（与主库 `data/ielts.sqlite` 分开）；长期画像落主库 `student_profile` / `grading_history` 表。

- 跑模块用 `python -m src.xxx.yyy`（如 `python -m src.rag.rubric`），脚本用 `python scripts/xxx.py`。
- 阶段 0 全程只用本地 embedding，**不调 DeepSeek**；改了 `.env` 的 key 后用 smoke test 验证连通。

## 开发阶段（先打通竖切，再加深）

- [阶段 0] 数据地基 ✅ **已完成**：异构数据归一化进 SQLite（2267 篇 silver）；rubric 按 (criterion×band) 切块、范文按 task×band×topic 分层灌 ChromaDB；剑桥 gold 已入库（51 sample 评测集 holdout + 21 model 锚点）。
- [阶段 1] 最薄竖切 ✅ **已完成**：最简 LangGraph（ingest→retrieve_rubric→四维顺序打分→aggregate）端到端出分；CLI `grade_essay.py` 出四维 band+依据+overall。
- [阶段 2] 把打分做准 ✅ **已完成**：eval harness（gold holdout 上算 MAE/±0.5/±1.0/QWK，temp=0 可复现，并发+超时）；范文锚定把 QWK 0.532→0.597（消融证实）；reflection 本测试集无增益、已从默认管道移除（代码保留）。默认打分管道 = 锚定开/reflection 关。
- [阶段 3] 工具 + 助手模式 + 成本路由 ✅ **已完成**：写作工具（vocab_upgrade/deconstruct_article/grammar_check/dictionary_lookup/exemplar_provide/score_predict）+ 扁平存库工具（save_vocab_entry/save_material_entry）；`create_react_agent` tool-calling 对话图 + CLI REPL；config 驱动成本路由（默认 flash）。**score_predict 复用阶段 2 打分管道**（测试锁死，无第二打分路径）。
- [阶段 4] 记忆与个性化 ✅ **已完成**：短期记忆用 LangGraph checkpointer（SQLite 后端，`thread_id` 隔离会话、跨进程断点续跑）；长期学生画像落 SQLite（episodic=band_history/grading_history 确定性 append；semantic=recurring_errors/weak_criteria/vocab_level）。`memory_write` 节点**增量蒸馏** semantic（只喂「旧画像 + 这一篇依据」，成本恒定）；批改会话外层图 `load_profile → grade → feedback → memory_write`（`src/graph/session.py`）。个性化只改**反馈措辞**，`grade` 节点只把 essay/task/prompt 喂进内层纯打分图，**profile 物理上进不了判分**（测试断言锁死）。回归实证：关个性化跑 gold holdout，QWK 0.597→0.605、±0.5 0.588→0.667（同一份打分代码的 API 抖动内，无向下漂移）。
- [阶段 5] 前端 + 词库/素材库 ✅ **已完成**：FastAPI 后端（`src/api/app.py`）包住现有图/工具/库——`POST /grade`（走 `build_grading_session_graph` 外层图，结构化 band+反馈）、`POST /chat`（SSE 流式，`build_assistant` + checkpointer，`thread_id`=会话 id）、`GET /lookup`、`GET/POST/DELETE /vocab`、`GET/POST/DELETE /materials`。纯 HTML/CSS/JS 五视图（批改/对话/查词/词库/素材库，`frontend/`），SSE 用 `fetch`+`ReadableStream` 逐 token 打字机渲染。**薄客户端**：智能全在后端，前端只 fetch+渲染，API key 只在后端 `.env`。护栏：打分/记忆逻辑零改动，Web 与 CLI 走同一套图。★坑★ `agent.stream(stream_mode="messages")` 会把**工具内部**的 LLM 调用（如 dictionary_lookup 的 call_json）也流出来，SSE 必须按 `meta["langgraph_node"]=="agent"` 过滤，只放行主节点 token。
- [阶段 6] 作品集打磨 ✅ **已完成**：可观测性——`src/obs/tracker.py` 用 passive LangChain callback 挂在唯一收口点 `get_llm`，自动记每次调用的档位/token/延迟/成本到 `data/logs/llm_calls.jsonl`（零改动打分/记忆/评测逻辑；`operation()` contextvar 只在胶水边界打标签；`OBS_LOG=0` 可关），`scripts/obs_summary.py` 按操作/档位汇总。README 重写（中文：pitch/架构图/Evaluation 严谨标注/设计决策/快速开始/复现边界）。依赖固定版本（`requirements.txt` 全 `==` + `requirements.lock` 全量），干净 venv 实测 20 passed。部署=本地（方案1）；上线（方案3）两拦路点（端点无鉴权会被刷账单、Ollama embedding 上云跑不动）记入记忆待将来。git 历史无密钥泄漏。

> **项目 v1 完成**（阶段 0–6 全部交付）。后续若做「上线」见记忆 deploy-online-future。

- [v1.1] 前端体验升级 ✅：暖色主题（CSS 变量 + Baskerville 衬线标题）；词库=生词本卡片墙（dictionary schema 加 pos/zh_def/ipa/双语例句，vocab 表迁移）；素材库=分类语料库（入库粒度根因修复：save_material_entry 逐条拆分、禁止整段存）；移除对话整段存按钮、存库走 agent 指令。★坑★ 改前端资源必 bump `index.html` 里 `?v=`（否则浏览器缓存看不到新版）。
- [v1.2] 作品集深度 + 卫生 ✅：**四维打分并行化**（fan-out/fan-in + dict reducer，只改调度不改打分逻辑；harness 实证 QWK 无向下漂移、单篇延迟 ~1.9×，见 docs/EVALUATION.md）；补齐三个设计承诺——**revision 改写示范**（外层图新节点，纯打分图/eval 不经过）、**助手滚动摘要**（pre_model_hook，超阈值压旧轮、只改喂 LLM 视图不落 checkpointer）、**对话/反馈 markdown 渲染**（mdLite：粗体/列表/引用/表格/标题）。删铁冗余 `save_to_library`。README 精简、Evaluation 详解迁 `docs/EVALUATION.md`。
- [v1.3] 检索层评测 ✅：`src/eval/retrieval.py` 给范文检索补分环节指标（Hit@1/Recall@5/MRR@10），27 条 holdout 题目查询（确定性采样、与 exemplar 语料零重叠、查询集落盘可人工校对）× 三档检索策略对比。**实证 metadata 过滤是必要设计**：纯向量召回同话题命中 ~34%（题目→范文正文是短查询对长文档的跨形态检索，话题信号弱，Task 1 尤甚），生产路径（task+topic 过滤 + 向量排序）按构造 100%。换 embedding 消融须重建索引再跑（跨向量空间检索无效）。详见 docs/EVALUATION.md。

- [v1.4] 池内向量选锚 ✅：修掉「ChromaDB 向量不进生产打分路径」的窟窿——范文锚点选取从 `_pick_spread`（band 均匀采样、池内不看话题）换成 `_pick_vector_spread`（过滤后的池**先按与本题 prompt 的向量相似度排序、每 band 取最贴的那篇、再跨 band 铺开**）。`run_cfg.anchor_rank` 分档（`spread`=旧行为/`vector`=新），harness 加 `anchored_vec_flash` 消融档。**单变量消融**（同 session 各 2 次，gold n=51 temp=0）：向量选锚 QWK 两次 {0.616, 0.611} 与基线 {0.579, 0.604} **不重叠**、均值 +0.022，MAE/±0.5 打平无退化 → 设为生产默认（`score.py`/`session.py` 的 `_GRADE_CFG` 加 `anchor_rank="vector"`）。这坐实了 v1.3 结论的另一半：过滤锚话题、向量在池内排序，分工都有量化证据。详见 docs/EVALUATION.md。★坑★ 选锚逻辑不改「怎么打分」，只改「喂哪几篇锚文」，故仍是那条被 eval 锁死的唯一打分管道。

- [v1.5] MCP server ✅：给 `src/tools/` 纯函数层加**第三种宿主**（继 CLI REPL、Web 后端之后）——`src/mcp_server.py` 用官方 `mcp` SDK 的 FastMCP，把 6 个**无状态分析工具**（score_predict/dictionary_lookup/grammar_check/vocab_upgrade/deconstruct_article/exemplar_provide）暴露成标准 MCP 服务，Claude Desktop / Cursor 等客户端可直接调用。**零重写智能**：薄适配器包已有纯函数，是「纯函数 + 框架适配」两层架构可复用性的活证据。刻意不暴露写库工具（save_vocab/save_material 需 user_id、写用户私有库，不适合匿名宿主）。stdio 传输（`python -m src.mcp_server`）；smoke 实测 6 工具注册正确 + dictionary_lookup 端到端调用通。★坑★ 文件名用 `mcp_server.py` 而非包 `src/mcp/`，避免 shadow `mcp` SDK 包。README 加接入配置；`mcp==1.28.1` 入 requirements。（注：score_predict 等调 DeepSeek，花服务持有者额度、勿公网裸暴露——同 deploy-online 记忆的鉴权隐患。）

> 每个阶段产出一个能跑的东西再进下一阶段。改动 scope 或决策前先和我确认。

## 明确不做

口语/听力/阅读、多智能体架构、模型微调、用户系统/鉴权、移动端原生 App。
