# IELTS Writing Agent

基于 LangGraph 的雅思写作（Task 1 + 2）批改与学习 Agent。
设计见 [`IELTS-Writing-Agent-设计文档.md`](IELTS-Writing-Agent-设计文档.md)，项目共识见 [`CLAUDE.md`](CLAUDE.md)。

## 快速开始

```powershell
# 1. 建虚拟环境并激活（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 装依赖
pip install -r requirements.txt

# 3. 配置密钥：复制模板后填入你的 DeepSeek key
copy .env.example .env
#   然后编辑 .env，填 DEEPSEEK_API_KEY=...

# 4. 拉本地 embedding 模型（首次）
ollama pull bge-m3

# 5. 跑连通性 smoke test，验证四条管线
python tests/test_connectivity.py
#   或： pytest tests/ -v
```

四项 smoke test 分别验证：DeepSeek API、本地 embedding、ChromaDB、SQLite。
未填 `DEEPSEEK_API_KEY` 时，DeepSeek 那项会 skip，其余三项仍应通过。

## 阶段 0：数据地基

把异构 CSV 归一化进 SQLite，官方 rubric 与精选范文灌进 ChromaDB（均为本地 embedding，**不需 DeepSeek key**）：

```powershell
python scripts/build_stage0.py     # 入库 SQLite + 构建两个 ChromaDB 集合
python scripts/demo_retrieve.py    # 验收：检索「环境话题 band7 Task2 范文」
pytest tests/test_stage0.py -v     # 断言测试
```

- silver 数据来源是网络数据集；gold 来源是剑桥真题 PDF（见下）。
- `essays` 表分 `train / holdout / exemplar` 三 split；topic 用关键词规则推断。
- ChromaDB 两集合：`rubric_descriptors`（按 criterion×band 结构感知切块）、`exemplar_essays`（按 task×band×topic 分层精选锚点）。

### 加入剑桥 gold（半自动：机器抽标签，人工转写手写正文）

剑桥真题里 band/考官评语/题目/范例是印刷体（机器确定性抽取），考生作文是手写（文本层乱码，需人工对照 PDF 转写）。

```powershell
python scripts/extract_gold.py        # 抽 9 本文本层书 → data/gold/manifest.json + bodies/*.txt（待填 stub）
#  → 打开 data/gold/bodies/*_sample.txt，对照 PDF 把分隔线下的乱码正文订正，改完把 STATUS 改成 DONE
python scripts/build_stage0.py        # 重建：已 DONE 的 sample 进 gold 评测集(holdout)，model 范文进锚点
```

- **STATUS 闸**：只有 `STATUS: DONE` 的 stub 才入库；model 范文默认 DONE（印刷体免转写），sample 默认 TODO。
- gold split：sample（真考生+官方 band）→ `holdout` 评测集（**不进锚点**，防泄漏）；model（范例，无 band）→ `exemplar` 高分锚点（band 记 9.0）。
- image-only 的 6 本（剑 9/14/15/16/18/19）需 OCR，推后。

## Evaluation（阶段 2）

评测纪律：**基准只用 gold tier holdout**（剑桥考官标注，n=51）；锚点全部来自 `split='exemplar'`，
跑前硬断言与 holdout 零重叠（防 few-shot 泄漏）；silver 只作四维小分的辅助参考。所有打分 **`temperature=0`** 以求可复现。

| 配置（gold holdout, n=51, temp=0） | MAE | ±0.5 | ±1.0 | **QWK** |
|---|---|---|---|---|
| baseline（flash，无锚定） | 0.667 | 64.7% | 86.3% | 0.532 |
| **anchored（flash + 范文 in-context 锚定）** | **0.637** | 58.8% | 86.3% | **0.597** |
| anchored + reflection（pro+thinking 回环） | 0.686 | 62.7% | 86.3% | 0.554 |

**锚定消融结论**：范文 in-context 锚定把 **QWK 从 0.532 提升到 0.597（+0.065）**、MAE 从 0.667 降到 0.637——
通过给模型「已知 band 的校准梯子」压掉大偏差、改善序数一致性。**reflection 回环在本测试集上无增益**（QWK 几乎不动、MAE 略升），故从默认管道移除。

> 为什么看 QWK：band 是序数量，QWK（Quadratic Weighted Kappa）用二次权重惩罚「差得远」的预测并校正随机一致，是自动作文评分（AES）文献的标准指标；纯准确率/±0.5 把所有错判等同、看不出偏离幅度。
> **方法论提醒**：钉 `temperature=0` 前，锚定「看似 +0.23 QWK」的大涨其实主要是采样方差——钉死随机性后如实测得 +0.065。可复现的数字才算数。

复现：`python -m src.eval.harness --config all`（写入 `data/eval/results.jsonl`）；`--compare` 看历史对比表。

## 技术栈

LangGraph · DeepSeek API（`v4-flash` / `v4-pro`）· 本地 bge-m3 embedding · ChromaDB · SQLite · FastAPI（后续）

## 进度

- [x] 地基：项目骨架 + 依赖 + 连通性 smoke test
- [x] 阶段 0：数据归一化入库 + rubric/范文灌 ChromaDB
- [x] 阶段 1：最薄竖切（LangGraph 四维打分跑通）
- [x] 阶段 2：eval harness + 锚定 + reflection + 消融（见上 Evaluation）
- [ ] 阶段 3+：工具/助手模式、记忆、前端——见设计文档 Roadmap
