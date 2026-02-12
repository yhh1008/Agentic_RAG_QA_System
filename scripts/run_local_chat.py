from __future__ import annotations

from src.agent.graph import run_agent


def main() -> None:
    print("Agentic RAG local chat, 输入 exit 退出")
    while True:
        query = input("\n你: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        result = run_agent(query)
        print(f"\n助手: {result['answer']}")
        if result["citations"]:
            print("\n引用:")
            for c in result["citations"]:
                print(f"- {c.get('doc_id')} / {c.get('chunk_id')} / {c.get('source')}")


if __name__ == "__main__":
    main()
