-- IELTS Writing Agent — SQLite schema（对应设计文档 §5.2）
-- 用 IF NOT EXISTS 让 init_db 可重复执行（幂等）。

-- 作文语料：归一化后的异构来源（gold/silver），小分与评语可空。
CREATE TABLE IF NOT EXISTS essays (
    id            INTEGER PRIMARY KEY,
    task_type     INTEGER,          -- 1 or 2
    prompt        TEXT,
    body          TEXT,
    overall_band  REAL,             -- nullable
    ta_band       REAL,             -- nullable
    cc_band       REAL,             -- nullable
    lr_band       REAL,             -- nullable
    gra_band      REAL,             -- nullable
    examiner_comment TEXT,          -- nullable
    source        TEXT,
    tier          TEXT,             -- 'gold' | 'silver'
    topic         TEXT,             -- 关键词规则推断
    split         TEXT,             -- 'train' | 'holdout' | 'exemplar'
    has_examiner_comment INTEGER DEFAULT 0,  -- 1=带真人考官评语（后续 gold 候选 / few-shot 锚点）
    justifications TEXT             -- JSON：d0 的四维 LLM 评分理由，nullable
);

CREATE INDEX IF NOT EXISTS idx_essays_lookup ON essays(task_type, overall_band, topic, split);

-- 词库（用户私有）
CREATE TABLE IF NOT EXISTS vocab_library (
    id              INTEGER PRIMARY KEY,
    user_id         TEXT,
    word            TEXT,
    context_sentence TEXT,
    alternatives    TEXT,           -- JSON
    nuance_note     TEXT,
    source_essay_id INTEGER,
    pos             TEXT,           -- v1.1 词性（如 'v. / n.'）
    zh_def          TEXT,           -- v1.1 中文释义
    en_def          TEXT,           -- v1.1 英文释义
    ipa             TEXT,           -- v1.1 音标
    examples        TEXT,           -- v1.1 JSON [{"en","zh"}]
    created_at      TIMESTAMP
);

-- 素材库（用户私有）
CREATE TABLE IF NOT EXISTS material_library (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT,
    type        TEXT,               -- v1.1 枚举：advanced_vocab | synonym | phrase
                                    --   | sentence_frame | outline | exemplar
    content     TEXT,               -- 单个条目本体（一个词/句式/范文，非整段回复）
    outline     TEXT,
    topic       TEXT,
    band        REAL,
    tags        TEXT,
    source_excerpt TEXT,            -- 出处原句
    note        TEXT,               -- v1.1 讲解/用法说明
    created_at  TIMESTAMP
);

-- 学生画像（长期记忆：episodic + semantic）
CREATE TABLE IF NOT EXISTS student_profile (
    user_id         TEXT PRIMARY KEY,
    recurring_errors TEXT,          -- JSON, semantic memory
    weak_criteria   TEXT,
    vocab_level     TEXT,
    band_history    TEXT,           -- JSON, episodic memory
    updated_at      TIMESTAMP
);

-- 批改历史
CREATE TABLE IF NOT EXISTS grading_history (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT,
    essay_id    INTEGER,
    scores      TEXT,               -- JSON
    feedback    TEXT,
    created_at  TIMESTAMP
);
