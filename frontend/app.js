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
  if (data.feedback) html += `<div class="feedback-box"><h3>个性化反馈</h3>${esc(data.feedback)}</div>`;
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
    // 从对话结果一键存素材库（存这条助手回复）。
    if (fullText.trim()) {
      const actions = document.createElement("div");
      actions.className = "msg-actions";
      const btn = document.createElement("button");
      btn.className = "small";
      btn.textContent = "＋ 存入素材库";
      btn.addEventListener("click", () => saveMaterial(fullText, btn));
      actions.appendChild(btn);
      botDiv.appendChild(actions);
    }
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
    let html = `<div class="dict-card"><div class="word">${esc(data.word)}</div>
      <strong>释义</strong><ul>${(data.definitions || []).map((d) => `<li>${esc(d)}</li>`).join("")}</ul>
      <strong>例句</strong><ul>${(data.examples || []).map((e) => `<li class="ex">${esc(e)}</li>`).join("")}</ul>
      <button class="small" id="save-word">＋ 存入词库</button></div>`;
    $("lookup-result").innerHTML = html;
    status.textContent = "";
    $("save-word").addEventListener("click", (ev) =>
      saveVocab(data.word, (data.examples || [])[0] || "", ev.target));
  } catch (e) {
    status.className = "status error"; status.textContent = "查询失败：" + e.message;
  }
}
$("lookup-btn").addEventListener("click", doLookup);
$("lookup-word").addEventListener("keydown", (e) => { if (e.key === "Enter") doLookup(); });

// ── 一键存库 ─────────────────────────────────────────────────────────
async function saveVocab(word, context, btn) {
  try {
    btn.disabled = true;
    await api("/vocab", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId(), word, context_sentence: context }),
    });
    btn.textContent = "✓ 已存入词库";
  } catch (e) { btn.disabled = false; alert("存词库失败：" + e.message); }
}

async function saveMaterial(content, btn) {
  try {
    btn.disabled = true;
    await api("/materials", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId(), type: "sentence_frame", content }),
    });
    btn.textContent = "✓ 已存入素材库";
  } catch (e) { btn.disabled = false; alert("存素材库失败：" + e.message); }
}

// ── 词库 / 素材库 列表 + 删除 ────────────────────────────────────────
async function loadVocab() {
  const box = $("vocab-list");
  box.innerHTML = "加载中…";
  try {
    const data = await api("/vocab?user_id=" + encodeURIComponent(userId()));
    if (!data.items.length) { box.innerHTML = '<div class="empty">词库还是空的。查词后点「存入词库」。</div>'; return; }
    box.innerHTML = data.items.map((it) => `
      <div class="lib-item"><div class="top">
        <span class="word">${esc(it.word)}</span>
        <button class="small danger" data-del="${it.id}">删除</button></div>
        ${it.context_sentence ? `<div class="sub">${esc(it.context_sentence)}</div>` : ""}
        <div class="meta">${esc(it.created_at)}</div></div>`).join("");
    box.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => delItem("/vocab/", b.dataset.del, loadVocab)));
  } catch (e) { box.innerHTML = `<div class="status error">加载失败：${esc(e.message)}</div>`; }
}

async function loadMaterials() {
  const box = $("materials-list");
  box.innerHTML = "加载中…";
  try {
    const data = await api("/materials?user_id=" + encodeURIComponent(userId()));
    if (!data.items.length) { box.innerHTML = '<div class="empty">素材库还是空的。对话里点「存入素材库」。</div>'; return; }
    box.innerHTML = data.items.map((it) => `
      <div class="lib-item"><div class="top">
        <span class="word">${esc(it.type)}${it.topic ? " · " + esc(it.topic) : ""}</span>
        <button class="small danger" data-del="${it.id}">删除</button></div>
        <div class="sub">${esc((it.content || "").slice(0, 300))}${(it.content || "").length > 300 ? "…" : ""}</div>
        <div class="meta">${esc(it.created_at)}</div></div>`).join("");
    box.querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => delItem("/materials/", b.dataset.del, loadMaterials)));
  } catch (e) { box.innerHTML = `<div class="status error">加载失败：${esc(e.message)}</div>`; }
}

async function delItem(base, id, reload) {
  if (!confirm("确认删除？")) return;
  try { await api(base + id, { method: "DELETE" }); reload(); }
  catch (e) { alert("删除失败：" + e.message); }
}

$("vocab-refresh").addEventListener("click", loadVocab);
$("materials-refresh").addEventListener("click", loadMaterials);
