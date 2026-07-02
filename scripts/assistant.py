"""阶段 3 CLI REPL：和 agentic 助手对话，验证 LLM 自主选 tool。

用法： python scripts/assistant.py     （输入 exit 退出）
每轮打印本轮调用了哪些 tool（供验证路由）。对话历史仅内存，不跨会话（记忆是阶段 4）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from src.agent.graph import build_assistant


def main() -> None:
    agent = build_assistant()
    history = []
    print("IELTS 写作助手 REPL — 输入 exit / quit 退出\n"
          "试试：『升级这句的 show：...』『拆解这篇文章：...』『delineate 什么意思』『给我打个分：...』")
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

        result = agent.invoke({"messages": history + [HumanMessage(user)]})
        msgs = result["messages"]
        turn = msgs[len(history):]                      # 本轮新增的消息
        tools = [tc["name"] for m in turn for tc in (getattr(m, "tool_calls", None) or [])]
        history = msgs

        if tools:
            print(f"  [调用工具: {', '.join(tools)}]")
        print(f"助手> {msgs[-1].content}")


if __name__ == "__main__":
    main()
