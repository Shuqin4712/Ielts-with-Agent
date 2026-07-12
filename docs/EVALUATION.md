# Evaluation — 评测方法与结果

> 本文是 IELTS Writing Agent 的评测细节（从 README 迁出，便于 README 保持精简）。
> 评测是本项目的核心差异化资产：**在考官标注的测试集上量化验证打分质量，而非自说自话。**

## 评测纪律（数字可信的前提）

- **Ground truth 只用 gold tier**：剑桥真题官方考官标注，`WHERE tier='gold' AND split='holdout'`，**n=51**。
- **零泄漏**：范文锚点全部来自 `split='exemplar'`，跑前硬断言与 holdout 零重叠（防 few-shot 泄漏）。
- **silver**（网络数据集，噪声大）只作辅助，**不作基准**。
- 所有打分 **`temperature=0`**，可复现。

## 主结果（overall band）

| 配置（gold holdout, n=51, temp=0） | MAE | ±0.5 | ±1.0 | **QWK** |
|---|---|---|---|---|
| baseline（flash，无锚定） | 0.667 | 64.7% | 86.3% | 0.532 |
| **anchored（flash + 范文 in-context 锚定）** | **0.637** | 58.8% | 86.3% | **0.597** |
| anchored + reflection（pro+thinking 回环） | 0.686 | 62.7% | 86.3% | 0.554 |

**锚定消融结论**：范文 in-context 锚定把 **QWK 0.532 → 0.597（+0.065）**、MAE 0.667 → 0.637——给模型"已知 band 的校准梯子"，压掉大偏差、改善序数一致性。

## 诚实的取舍

- **reflection 回环在本测试集上无增益**（QWK 反降、MAE 略升），故从默认管道移除（代码保留）。默认打分管道 = 锚定开 / reflection 关。
- **方法论提醒**：钉 `temperature=0` 之前，锚定"看似 +0.23 QWK"的大涨其实主要是采样方差；钉死随机性后如实测得 **+0.065**。可复现的数字才算数。

> **为什么看 QWK**：band 是序数量，QWK（Quadratic Weighted Kappa）用二次权重惩罚"差得远"的预测并校正随机一致，是自动作文评分（AES）文献的标准指标；纯准确率把所有错判等同、看不出偏离幅度。

## 记忆没污染打分

关掉个性化跑同一份 gold holdout，QWK 0.597→0.605、±0.5 0.588→0.667（同代码 temp=0 的 API 抖动内，无向下漂移）——佐证画像只改反馈措辞、进不了判分。

## 四维打分并行化不影响质量（v1.2）

把四维打分从串行改成 LangGraph fan-out/fan-in 并行后，用**同一 harness** 在 gold holdout 上验证打分质量不变：

| anchored_flash 运行 | 调度 | MAE | ±0.5 | QWK |
|---|---|---|---|---|
| 改前 #1 | 串行 | 0.637 | 0.588 | 0.597 |
| 改前 #2 | 串行 | 0.598 | 0.667 | 0.605 |
| 改后 #1 | 并行 | 0.686 | 0.588 | 0.585 |
| 改后 #2 | 并行 | 0.637 | 0.647 | 0.624 |

两次并行运行 QWK {0.585, 0.624} **bracket** 两次串行 {0.597, 0.605}，并行均值 0.605 ≥ 串行均值 0.601——**无系统性下移**，差异是 temp=0 的 API 抖动（4 样本显示 QWK 在 ~0.60 ± 0.02）。单篇批改延迟：四维之和 ~25.5s → 并行墙钟 ~13.3s（**~1.9×**）。**调度优化与打分质量解耦。**

## 检索层评测：metadata 过滤不是装饰（v1.3）

端到端 QWK 只能证明「检索 + 锚定」整体有价值，定位不了检索环节自身的损耗，
故给范文检索补一个**分环节指标**：给一道 holdout 题目（prompt），召回结果里
同 (task_type, topic) 的范文算相关，量化 Hit@1 / Recall@5 / MRR@10。

**纪律**：查询集 n=27（每个库存 ≥5 的 task×topic 格确定性采样 1 题，seed 固定），
全部来自 `split='holdout'`，与被检索语料（`split='exemplar'`）零重叠；
查询集落盘为 `data/eval/retrieval_queries.jsonl`，**人工可校对**——它就是标注文件。

| 检索策略（n=27, bge-m3） | Hit@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| vector（纯向量，无过滤） | 0.370 | 0.341 | 0.513 |
| vector_task（task 过滤 + 向量） | 0.407 | 0.378 | 0.551 |
| **filtered（生产路径：task+topic 过滤 + 向量）** | **1.000** | **1.000** | **1.000** |

**结论**：纯向量召回的同话题命中率只有 ~34%——「题目 → 范文正文」是**短查询对长文档的
跨形态检索**，embedding 相似度大量被写作风格/文体信号占据，话题信号弱。结构感知
RAG 的 metadata 过滤把话题命中锚死在 100%（按构造），**过滤是必要设计而非装饰**；
向量相似度在过滤后的池内只负责排序。

**错例分析**：vector 档最差的查询全是 Task 1（Task 1 MRR 0.424 vs Task 2 0.585）——
图表题题干是「The chart shows...」式描述，话题词汇稀薄，向量召回天然吃亏；
Task 2 议论文题干携带完整话题语义，表现明显更好。

**局限**：相关性标注 = 规则打的 topic 标签（弱标注，`filtered` 档的 1.000 是按构造成立的
sanity 上限，不是「检索完美」）；n=27 小样本。换 embedding 做消融时须用新模型**重建索引**
到独立集合再跑（查询侧单边换模型 = 跨向量空间检索，结果无效）。

```powershell
python -m src.eval.retrieval                    # 只用本地 embedding，不调 DeepSeek
pytest tests/test_retrieval_eval.py -v          # 指标纯函数 + 查询集契约（LLM-free）
```

## 池内向量选锚 > band 均匀采样（v1.4）

**动机（把检索窟窿变亮点）**：v1.3 检索评测证明了 metadata 过滤锚话题的价值，但生产打分路径此前对**过滤后的池**只做 band 均匀采样（`_pick_spread`：按 band 排序取均匀下标，池内不看话题相关性），ChromaDB 的**向量相似度根本没进打分路径**——只在检索评测里用。v1.4 让向量真正上生产：过滤后的池**先按与本题 prompt 的向量相似度排序，每个 band 留最贴的那篇，再跨 band 铺开**（`_pick_vector_spread`），既保留校准梯子的跨 band 结构，又让每一档锚文话题/文体贴近被评作文。

**单变量消融**（`anchored_vec_flash` vs `anchored_flash`，唯一差异 = 选锚策略；同 session 背靠背各 2 次，控 API 抖动；gold holdout n=51, temp=0）：

| 选锚策略 | QWK（两次） | MAE（两次） | ±0.5（两次） |
|---|---|---|---|
| band 均匀采样（`_pick_spread`，v1.2 基线） | 0.579 / 0.604 | 0.598 / 0.598 | 0.686 / 0.706 |
| **池内向量排序（`_pick_vector_spread`，生产默认）** | **0.616 / 0.611** | 0.608 / 0.618 | 0.706 / 0.647 |

**结论**：向量选锚的 QWK 两次 **{0.611, 0.616} 与均匀采样 {0.579, 0.604} 两次不重叠**（向量下界 0.611 > 均匀上界 0.604），均值 **+0.022**——在序数一致性（本项目头号指标）上有**小而一致**的增益，比单次数字更可信。MAE 打平（均匀微优 0.015，落在自身历史 0.598–0.686 抖动内）、±0.5 打平。既无退化、QWK 有增益、且锚点梯子结构更好（去重后跨 band 更完整），故设为**生产默认**（`score.py` / `session.py` 的 `anchor_rank="vector"`）。

**意义**：这是「向量库不再是装饰」的实证——过滤锚话题、向量在池内排序，两者分工都有量化证据。**局限**：+0.022 仍是小增益、n=51；换 embedding 须重建索引再测（同 v1.3）。

```powershell
python -m src.eval.harness --config anchored_vec_flash --no-fourdim   # 向量选锚
python -m src.eval.harness --config anchored_flash --no-fourdim       # band 均匀采样对照
```

## 已知局限（诚实披露）

- **四维小分未评测**：gold holdout 只有官方 overall band，无 TA/CC/LR/GRA 小分标注（`has_sub=0`），故小分准确度**无法量化**，只有 overall 有 ground truth。
- **样本量小（n=51）**：置信区间较宽；上面的 QWK 差异应理解为"在小样本 + API 抖动内"，而非精确点估计。
- **单用户 demo**：无鉴权 / 无速率限制，不适合公网直接暴露（见 CLAUDE.md「明确不做」）。

## 复现

```powershell
python -m src.eval.harness --config all      # 全配置跑，写 data/eval/results.jsonl
python -m src.eval.harness --compare         # 打印历史对比表
```
