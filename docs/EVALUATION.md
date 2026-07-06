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

## 已知局限（诚实披露）

- **四维小分未评测**：gold holdout 只有官方 overall band，无 TA/CC/LR/GRA 小分标注（`has_sub=0`），故小分准确度**无法量化**，只有 overall 有 ground truth。
- **样本量小（n=51）**：置信区间较宽；上面的 QWK 差异应理解为"在小样本 + API 抖动内"，而非精确点估计。
- **单用户 demo**：无鉴权 / 无速率限制，不适合公网直接暴露（见 CLAUDE.md「明确不做」）。

## 复现

```powershell
python -m src.eval.harness --config all      # 全配置跑，写 data/eval/results.jsonl
python -m src.eval.harness --compare         # 打印历史对比表
```
