"""阶段 3/4 CLI REPL：和 agentic 助手对话，验证 LLM 自主选 tool + 短期记忆。

用法：
  python scripts/assistant.py                 # 新会话（自动生成 thread_id）
  python scripts/assistant.py --thread demo1   # 指定 thread_id，可跨进程续跑同一会话
  python scripts/assistant.py --no-memory       # 关短期记忆（阶段 3 单轮行为）

每轮打印本轮调用了哪些 tool（供验证路由）。开短期记忆时，多轮对话由 LangGraph
checkpointer 按 thread_id 接续——后一轮记得前一轮说过的内容，且落盘可断点续跑。
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from src.agent.graph import build_assistant
from src.memory.checkpoint import get_checkpointer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread", help="会话 thread_id（省略则随机生成一个新会话）")
    ap.add_argument("--no-memory", action="store_true", help="关短期记忆（每轮独立）")
    args = ap.parse_args()

    use_memory = not args.no_memory
    checkpointer = get_checkpointer() if use_memory else None
    agent = build_assistant(checkpointer=checkpointer)

    thread_id = args.thread or uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": thread_id}} if use_memory else {}

    print("IELTS 写作助手 REPL — 输入 exit / quit 退出")
    if use_memory:
        print(f"短期记忆：开（thread_id={thread_id}，多轮接续、落盘可续跑）")
    else:
        print("短期记忆：关（每轮独立）")
    print("试试：『升级这句的 show：...』『拆解这篇文章：...』『delineate 什么意思』『给我打个分：...』")

    # 无记忆时才需在客户端手动累加历史；有记忆时只传新消息，历史由 checkpointer 恢复。
    history = []
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            break

        if use_memory:
            # 有记忆：只喂新消息，返回的是全线程消息（含历史）；本轮 = 末尾那条
            # HumanMessage 起到最后（用于抽本轮调用的 tool）。
            all_msgs = agent.invoke({"messages": [HumanMessage(user)]}, cfg)["messages"]
            turn = all_msgs[_last_human_idx(all_msgs):]
        else:
            all_msgs = agent.invoke({"messages": history + [HumanMessage(user)]})["messages"]
            turn = all_msgs[len(history):]
            history = all_msgs

        tools = [tc["name"] for m in turn for tc in (getattr(m, "tool_calls", None) or [])]
        if tools:
            print(f"  [调用工具: {', '.join(tools)}]")
        print(f"助手> {all_msgs[-1].content}")


def _last_human_idx(msgs) -> int:
    """最后一条 HumanMessage 的下标（本轮从这里开始）。"""
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            return i
    return 0


if __name__ == "__main__":
    main()
