// 薄客户端：只做 fetch + 渲染，所有智能在后端。API 同源，base 留空。
"use strict";

// ── user_id（demo 用，落 localStorage）──────────────────────────────
const userInput = document.getElementById("user_id");
userInput.value = localStorage.getItem("ielts_user") || "demo";
userInput.addEventListener("change", () => {
  localStorage.setItem("ielts_user", userInput.value.trim() || "demo");
  loadVocab(); loadMaterials();  // 换用户 → 刷新库视图
});
const userId = () => userInput.value.trim() || "demo";

// ── tab 切换 ─────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    document.getElementById("view-" + view).classList.add("active");
    if (view === "vocab") loadVocab();
    if (view === "materials") loadMaterials();
  });
});

// ── 小工具 ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const esc = (s) => (s ?? "").toString()
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

async function api(path, opts = {}) {
  // 统一错误处理：非 2xx 或坏 JSON 都抛出可读信息，调用方 catch 后提示、不白屏。
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; }
  catch { throw new Error("后端返回了非 JSON：" + text.slice(0, 120)); }
  if (!res.ok || data.error) throw new Error(data.error || data.detail || ("HTTP " + res.status));
  return data;
}

// ── 批改 ─────────────────────────────────────────────────────────────
$("grade-btn").addEventListener("click", async () => {
  const essay = $("grade-essay").value.trim();
  const status = $("grade-status");
  const result = $("grade-result");
  if (!essay) { status.textContent = "请先粘贴作文。"; status.className = "status error"; return; }
  $("grade-btn").disabled = true;
  status.className = "status"; status.textContent = "批改中…（走四维打分 + 反馈，需十几秒）";
  result.innerHTML = "";
  try {
    const data = await api("/grade", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId(), essay,
        task_type: parseInt($("grade-task").value, 10),
        prompt: $("grade-prompt").value.trim(),
      }),
    });
    renderGrade(data);
    status.textContent = "";
  } catch (e) {
    status.className = "status error"; status.textContent = "批改失败：" + e.message;
  } finally {
    $("grade-btn").disabled = false;
  }
});

function renderGrade(data) {
  const names = { TA: "任务回应 TA", CC: "连贯衔接 CC", LR: "词汇资源 LR", GRA: "语法 GRA" };
  let html = "";
  for (const c of ["TA", "CC", "LR", "GRA"]) {
    const s = data.dimension_scores[c];
    html += `<div class="crit-card">
      <div class="crit-head"><span class="crit-name">${names[c]}</span>
        <span class="band-pill">band ${s.band}</span></div>
      <div class="crit-ev">${esc(s.evidence)}</div></div>`;
  }
  html += `<div class="crit-card"><div class="crit-head">
      <span class="crit-name">Overall</span>
      <span class="band-pill overall-pill">band ${data.overall_band}</span></div></div>`;
  if (data.feedback) html += `<div class="feedback-box"><h3>个性化反馈</h3><div class="md">${mdLite(data.feedback)}</div></div>`;
  const revs = data.revision || [];
  if (revs.length) {
    html += `<div class="feedback-box"><h3>改写示范（针对最弱维度）</h3>${revs.map((r) => `
      <div class="rev-item">
        <div class="rev-orig">${esc(r.original || "")}</div>
        <div class="rev-new">${esc(r.revised || "")}</div>
        ${r.why ? `<div class="rev-why">💡 ${esc(r.why)}</div>` : ""}
      </div>`).join("")}</div>`;
  }
  $("grade-result").innerHTML = html;
}

// ── 对话（SSE 流式）──────────────────────────────────────────────────
let convId = "conv-" + Math.random().toString(36).slice(2, 10);
const chatLog = $("chat-log");

$("chat-new").addEventListener("click", () => {
  convId = "conv-" + Math.random().toString(36).slice(2, 10);
  chatLog.innerHTML = "";
});

function addMsg(who, text) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.innerHTML = `<div class="who">${who === "user" ? "你" : "助手"}</div>
    <div class="bubble"></div>`;
  div.querySelector(".bubble").textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

async function sendChat() {
  const box = $("chat-msg");
  const msg = box.value.trim();
  if (!msg) return;
  box.value = "";
  $("chat-send").disabled = true;
  addMsg("user", msg);
  const botDiv = addMsg("assistant", "");
  const bubble = botDiv.querySelector(".bubble");

  try {
    const res = await fetch("/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId(), conversation_id: convId, message: msg }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);

    // 读 SSE：按 \n\n 切帧，每帧解析 data: {...}。逐 token 追加实现打字机效果。
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", fullText = "", tools = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();  // 末尾可能是半帧，留到下轮
      for (const f of frames) {
        const line = f.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const evt = JSON.parse(line.slice(5).trim());
        if (evt.type === "token") {
          fullText += evt.text; bubble.textContent = fullText;
          chatLog.scrollTop = chatLog.scrollHeight;
        } else if (evt.type === "tools") {
          tools = evt.tools;
        } else if (evt.type === "error") {
          throw new Error(evt.message);
        }
      }
    }
    if (tools.length) {
      const chip = document.createElement("span");
      chip.className = "tools-chip";
      chip.textContent = "🔧 " + tools.join(", ");
      botDiv.querySelector(".who").appendChild(chip);
    }
    // 流式期间是纯文本打字机；收到 done 后一次性把终稿渲染成 markdown
    // （流式中途转会闪烁，故只在最后转）。mdLite 已 escape 防 XSS。
    if (fullText.trim()) bubble.innerHTML = `<div class="md">${mdLite(fullText)}</div>`;
    // v1.1：不再提供「整段回复存一条」按钮（那是素材库脏数据的根源）。
    // 存库走 agent 指令：用户说「帮我把这些存进素材库/词库」，agent 逐条拆分入库。
  } catch (e) {
    bubble.textContent = "⚠️ 对话失败：" + e.message;
  } finally {
    $("chat-send").disabled = false;
  }
}

$("chat-send").addEventListener("click", sendChat);
$("chat-msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// ── 查词 ─────────────────────────────────────────────────────────────
async function doLookup() {
  const word = $("lookup-word").value.trim();
  const status = $("lookup-status");
  if (!word) { status.textContent = "请输入单词。"; status.className = "status error"; return; }
  status.className = "status"; status.textContent = "查询中…";
  $("lookup-result").innerHTML = "";
  try {
    const data = await api("/lookup?word=" + encodeURIComponent(word));
    // v1.1 schema：ipa / pos / zh_def / en_def / examples[{en,zh}]（旧字段缺失时降级）
    const exs = (data.examples || []).map((e) =>
      typeof e === "string" ? { en: e, zh: "" } : e);
    let html = `<div class="dict-card">
      <div class="dict-head">
        <span class="word">${esc(data.word)}</span>
        ${data.ipa ? `<span class="ipa">/${esc(data.ipa)}/</span>` : ""}
        ${data.pos ? `<span class="pos-chip">${esc(data.pos)}</span>` : ""}
      </div>
      ${data.zh_def ? `<div class="zh-def">${esc(data.zh_def)}</div>` : ""}
      ${data.en_def ? `<div class="en-def">${esc(data.en_def)}</div>` : ""}
      ${exs.length ? `<div class="ex-block">${exs.map((e) => `
        <div class="ex-pair"><div class="ex">${esc(e.en)}</div>
        ${e.zh ? `<div class="ex-zh">${esc(e.zh)}</div>` : ""}</div>`).join("")}</div>` : ""}
      <button class="small" id="save-word">＋ 存入词库</button></div>`;
    $("lookup-result").innerHTML = html;
    status.textContent = "";
    $("save-word").addEventListener("click", (ev) => saveVocab({
      word: data.word, context_sentence: exs[0]?.en || "",
      pos: data.pos || null, zh_def: data.zh_def || null,
      en_def: data.en_def || null, ipa: data.ipa || null, examples: exs,
    }, ev.target));
  } catch (e) {
    status.className = "status error"; status.textContent = "查询失败：" + e.message;
  }
}
$("lookup-btn").addEventListener("click", doLookup);
$("lookup-word").addEventListener("keydown", (e) => { if (e.key === "Enter") doLookup(); });

// ── 一键存库 ─────────────────────────────────────────────────────────
async function saveVocab(entry, btn) {
  // entry: {word, context_sentence, pos?, zh_def?, en_def?, ipa?, examples?}
  try {
    btn.disabled = true;
    await api("/vocab", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId(), ...entry }),
    });
    btn.textContent = "✓ 已存入词库";
  } catch (e) { btn.disabled = false; alert("存词库失败：" + e.message); }
}

// ── 词库：生词本卡片墙（搜索 / 排序 / 字母分组 / 点击展开） ──────────
let vocabItems = [];

async function loadVocab() {
  const box = $("vocab-list");
  box.innerHTML = "加载中…";
  try {
    const data = await api("/vocab?user_id=" + encodeURIComponent(userId()));
    vocabItems = data.items;
    renderVocab();
  } catch (e) { box.innerHTML = `<div class="status error">加载失败：${esc(e.message)}</div>`; }
}

function renderVocab() {
  const box = $("vocab-list");
  const q = $("vocab-search").value.trim().toLowerCase();
  const sort = $("vocab-sort").value;

  if (!vocabItems.length) {
    box.innerHTML = '<div class="empty">生词本还是空的——在「查词」或对话里收藏，生词会出现在这里。</div>';
    return;
  }
  let items = vocabItems.filter((it) => !q ||
    [it.word, it.zh_def, it.en_def, it.context_sentence]
      .some((f) => (f || "").toLowerCase().includes(q)));
  if (!items.length) { box.innerHTML = '<div class="empty">没有匹配的生词。</div>'; return; }

  if (sort === "alpha") {
    items = [...items].sort((a, b) => (a.word || "").toLowerCase()
      .localeCompare((b.word || "").toLowerCase()));
  } else {
    items = [...items].sort((a, b) => b.id - a.id);   // 新添加在前
  }

  let html = "", lastLetter = "";
  for (const it of items) {
    if (sort === "alpha") {                            // 字母分组头
      const letter = ((it.word || "?")[0] || "?").toUpperCase();
      if (letter !== lastLetter) {
        html += `<div class="letter-head">${esc(letter)}</div>`;
        lastLetter = letter;
      }
    }
    html += vocabCard(it);
  }
  box.innerHTML = html;

  box.querySelectorAll(".vocab-card").forEach((card) => {
    card.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;          // 按钮点击不触发折叠
      card.classList.toggle("open");
    });
  });
  box.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", () => delItem("/vocab/", b.dataset.del, loadVocab)));
}

function vocabCard(it) {
  let exs = [];
  try { exs = JSON.parse(it.examples || "[]") || []; } catch { exs = []; }
  return `
  <div class="lib-item vocab-card">
    <div class="vc-head">
      <span class="word">${esc(it.word)}</span>
      ${it.ipa ? `<span class="ipa">/${esc(it.ipa)}/</span>` : ""}
      ${it.pos ? `<span class="pos-chip">${esc(it.pos)}</span>` : ""}
      <span class="vc-caret">›</span>
    </div>
    ${it.zh_def ? `<div class="zh-def">${esc(it.zh_def)}</div>` : ""}
    <div class="vc-detail">
      ${it.en_def ? `<div class="en-def">${esc(it.en_def)}</div>` : ""}
      ${exs.map((e) => `<div class="ex-pair"><div class="ex">${esc(e.en || e)}</div>
        ${e.zh ? `<div class="ex-zh">${esc(e.zh)}</div>` : ""}</div>`).join("")}
      ${it.context_sentence ? `<div class="ex-pair"><div class="ex">${esc(it.context_sentence)}</div>
        <div class="ex-zh">收藏时的语境句</div></div>` : ""}
      ${it.nuance_note ? `<div class="note">💡 ${esc(it.nuance_note)}</div>` : ""}
      <div class="vc-actions">
        <span class="meta">${esc((it.created_at || "").slice(0, 10))}</span>
        <button class="small danger" data-del="${it.id}">删除</button>
      </div>
    </div>
  </div>`;
}

// ── 素材库：分类语料库（chips 筛选 / 搜索 / 展开 / 复制） ────────────
const MAT_TYPES = {
  advanced_vocab: "高级词汇", synonym: "同义替换", phrase: "短语",
  sentence_frame: "句式模板", outline: "思路提纲", exemplar: "范文",
};
let materialItems = [], matFilter = "all";

// 轻量 markdown 渲染：先 escape 防 XSS，再处理粗体/行内代码/列表/引用/表格。
// 表格常见于词汇升级回复（替换词×语域×band 对照），故 demo 里值得渲染好。
function mdLite(text) {
  const lines = esc(text || "").split(/\r?\n/);
  const inline = (s) => s
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const isSep = (l) => l.includes("-") && /^\s*\|?[\s:|-]+\|?\s*$/.test(l);
  const isRow = (l) => /^\s*\|.*\|/.test(l);

  let html = "", inList = false, i = 0;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };

  while (i < lines.length) {
    const line = lines[i].trimEnd();
    // 表格：一行 | … | 后紧跟分隔行 |---|---|
    if (isRow(line) && i + 1 < lines.length && isSep(lines[i + 1])) {
      closeList();
      const head = cells(line).map((c) => `<th>${inline(c)}</th>`).join("");
      i += 2;
      let body = "";
      while (i < lines.length && isRow(lines[i])) {
        body += `<tr>${cells(lines[i]).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`;
        i++;
      }
      html += `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); const lvl = Math.min(h[1].length + 2, 4); html += `<h${lvl}>${inline(h[2])}</h${lvl}>`; i++; continue; }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { closeList(); html += "<hr>"; i++; continue; }
    const li = line.match(/^\s*[-*]\s+(.*)$/);
    if (li) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(li[1])}</li>`;
      i++; continue;
    }
    closeList();
    if (/^\s*>\s?/.test(line)) html += `<blockquote>${inline(line.replace(/^\s*>\s?/, ""))}</blockquote>`;
    else if (line.trim()) html += `<p>${inline(line)}</p>`;
    i++;
  }
  closeList();
  return html;
}

// 句式模板：等宽 + 高亮 X/Y/Z 占位符
function frameHtml(text) {
  const body = esc(text || "").replace(/\b([XYZ])\b/g, '<span class="ph">$1</span>');
  return `<div class="frame">${body}</div>`;
}

async function loadMaterials() {
  const box = $("materials-list");
  box.innerHTML = "加载中…";
  try {
    const data = await api("/materials?user_id=" + encodeURIComponent(userId()));
    materialItems = data.items;
    renderMatChips();
    renderMaterials();
  } catch (e) { box.innerHTML = `<div class="status error">加载失败：${esc(e.message)}</div>`; }
}

function renderMatChips() {
  const counts = {};
  for (const it of materialItems) counts[it.type] = (counts[it.type] || 0) + 1;
  let html = `<button class="chip${matFilter === "all" ? " active" : ""}" data-f="all">全部 (${materialItems.length})</button>`;
  for (const [t, label] of Object.entries(MAT_TYPES)) {
    if (!counts[t]) continue;
    html += `<button class="chip${matFilter === t ? " active" : ""}" data-f="${t}">${label} (${counts[t]})</button>`;
  }
  $("materials-chips").innerHTML = html;
  $("materials-chips").querySelectorAll(".chip").forEach((c) =>
    c.addEventListener("click", () => { matFilter = c.dataset.f; renderMatChips(); renderMaterials(); }));
}

function renderMaterials() {
  const box = $("materials-list");
  const q = $("materials-search").value.trim().toLowerCase();
  if (!materialItems.length) {
    box.innerHTML = '<div class="empty">素材库还是空的——在对话里让助手帮你存：如「把这些句式存进素材库」。</div>';
    return;
  }
  const items = materialItems
    .filter((it) => matFilter === "all" || it.type === matFilter)
    .filter((it) => !q || [it.content, it.note, it.topic, it.source_excerpt]
      .some((f) => (f || "").toLowerCase().includes(q)))
    .sort((a, b) => b.id - a.id);
  if (!items.length) { box.innerHTML = '<div class="empty">没有匹配的素材。</div>'; return; }
  box.innerHTML = items.map(materialCard).join("");

  box.querySelectorAll(".mat-card").forEach((card) => {
    card.addEventListener("click", (e) => {
      if (e.target.closest("button")) return;
      card.classList.toggle("open");
    });
  });
  box.querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", () => delItem("/materials/", b.dataset.del, loadMaterials)));
  box.querySelectorAll("[data-copy]").forEach((b) =>
    b.addEventListener("click", async () => {
      const it = materialItems.find((x) => x.id === +b.dataset.copy);
      try { await navigator.clipboard.writeText(it?.content || ""); b.textContent = "✓ 已复制"; }
      catch { b.textContent = "复制失败"; }
      setTimeout(() => { b.textContent = "复制"; }, 1500);
    }));
}

function materialCard(it) {
  const label = MAT_TYPES[it.type] || it.type;   // 旧枚举值显示原文（降级）
  const long = (it.content || "").length > 300;
  const body = it.type === "sentence_frame" ? frameHtml(it.content)
    : `<div class="md">${mdLite(it.content)}</div>`;
  const hasDetail = it.note || it.source_excerpt || it.topic || it.band;
  return `
  <div class="lib-item mat-card${long ? " clamped" : ""}">
    <div class="top">
      <span class="mat-badge">${esc(label)}</span>
      <span class="mat-tools">
        <button class="small" data-copy="${it.id}">复制</button>
        <span class="vc-caret">›</span>
      </span>
    </div>
    <div class="mat-body">${body}</div>
    ${long ? '<div class="fade-hint">点击展开全文</div>' : ""}
    <div class="mat-detail">
      ${it.note ? `<div class="note">💡 ${esc(it.note)}</div>` : ""}
      ${it.source_excerpt ? `<div class="ex-pair"><div class="ex">${esc(it.source_excerpt)}</div>
        <div class="ex-zh">出处原句</div></div>` : ""}
      <div class="vc-actions">
        <span class="meta">${it.topic ? esc(it.topic) + " · " : ""}${it.band ? "band " + esc(it.band) + " · " : ""}${esc((it.created_at || "").slice(0, 10))}</span>
        <button class="small danger" data-del="${it.id}">删除</button>
      </div>
    </div>
    ${!hasDetail && !long ? "" : ""}
  </div>`;
}

async function delItem(base, id, reload) {
  if (!confirm("确认删除？")) return;
  try { await api(base + id, { method: "DELETE" }); reload(); }
  catch (e) { alert("删除失败：" + e.message); }
}

$("vocab-refresh").addEventListener("click", loadVocab);
$("vocab-search").addEventListener("input", renderVocab);
$("vocab-sort").addEventListener("change", renderVocab);
$("materials-refresh").addEventListener("click", loadMaterials);
$("materials-search").addEventListener("input", renderMaterials);
