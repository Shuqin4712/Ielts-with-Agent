"""轻量关键词规则的话题标注。

两个 CSV 都没有 topic 列，但「按话题检索范文」需要它。阶段 0 先用
关键词规则零成本打标签：在题目(prompt)上匹配各话题的关键词，命中最多者胜，
都不中则 'general'。不够精细的后续再上 LLM 分类（见 DESIGN Roadmap）。
"""
from __future__ import annotations

import re

# 话题 → 关键词列表（雅思 Task 2 常见母题 + Task 1 图表常见主题）。
# 关键词用小写、词边界匹配，尽量选区分度高的词。
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "environment": ["environment", "environmental", "pollution", "climate", "carbon",
                    "global warming", "recycl", "renewable", "wildlife", "ecolog",
                    "emission", "sustainab", "deforestation", "energy"],
    "education": ["education", "school", "student", "university", "teacher", "learn",
                  "study", "academic", "classroom", "curriculum", "exam", "homework"],
    "technology": ["technolog", "internet", "computer", "smartphone", "online", "digital",
                   "social media", "artificial intelligence", "robot", "device", "software"],
    "health": ["health", "medical", "disease", "diet", "obesity", "exercise", "hospital",
               "doctor", "mental health", "fitness", "smoking", "nutrition"],
    "crime": ["crime", "criminal", "prison", "punishment", "police", "law", "offender",
              "justice", "theft", "violence"],
    "government": ["government", "policy", "tax", "public spending", "politic", "state",
                   "authorities", "law", "regulation", "vote", "election"],
    "work": ["work", "job", "employee", "employer", "career", "salary", "workplace",
             "unemployment", "profession", "business", "office"],
    "family": ["family", "child", "children", "parent", "elderly", "marriage",
               "generation", "raising", "household"],
    "globalisation": ["globalis", "globaliz", "international", "culture", "tradition",
                      "immigrant", "migration", "multicultural", "foreign"],
    "media": ["media", "news", "advertis", "newspaper", "television", "journalis",
              "celebrity", "magazine"],
    "transport": ["transport", "traffic", "car", "public transport", "vehicle", "road",
                  "commut", "railway", "airport"],
    "tourism": ["tourism", "tourist", "travel", "holiday", "visit", "destination"],
    "city": ["city", "cities", "urban", "rural", "housing", "population", "countryside",
             "town", "infrastructure"],
    "money": ["money", "economic", "economy", "income", "wealth", "poverty", "consumer",
              "spending", "financial", "cost"],
    "sport": ["sport", "athlete", "olympic", "competition", "football", "team"],
}


def _compile(words: list[str]) -> list[re.Pattern]:
    # 用 \b 词边界但允许词干（如 "recycl" 匹配 recycling）。
    return [re.compile(r"\b" + re.escape(w), re.IGNORECASE) for w in words]


_COMPILED = {topic: _compile(words) for topic, words in TOPIC_KEYWORDS.items()}


def infer_topic(prompt: str, essay: str = "") -> str:
    """返回命中关键词最多的话题；都不中则 'general'。

    主要看题目；题目信号弱时用正文兜底（权重低）。
    """
    text_q = prompt or ""
    text_e = essay or ""
    best, best_score = "general", 0
    for topic, patterns in _COMPILED.items():
        score = sum(2 for p in patterns if p.search(text_q))      # 题目命中权重 2
        score += sum(1 for p in patterns if p.search(text_e))     # 正文命中权重 1
        if score > best_score:
            best, best_score = topic, score
    return best
