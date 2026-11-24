"""Прямой запрос к Ollama без агента и RAG."""
import asyncio

from rag_agent.core.config import get_settings
from rag_agent.integrations.ollama.client import OllamaClient
from rag_agent.services.llm_service import LLMService


async def main():
    settings = get_settings()
    client = OllamaClient(str(settings.ollama_base_url), settings.ollama_timeout)
    try:
        service = LLMService(client, settings)
        print(await service.generate("Кратко представься на русском языке."))
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
