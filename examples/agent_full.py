"""Ручной чат с RAG; Ollama запускается отдельно."""
import asyncio

from rag_agent.bootstrap import open_services


async def main():
    async with open_services() as services:
        agent = await services.get_agent()
        print("LocalMind: задайте вопрос, exit — выход.")
        while True:
            try:
                message = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message.lower() in {"exit", "quit"}:
                break
            if message:
                result = await agent.ainvoke(message)
                print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
