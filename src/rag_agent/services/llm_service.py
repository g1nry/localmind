from rag_agent.services.embedding_service import EmbeddingService


class LLMService(EmbeddingService):
    """Совместимый сервис генерации для старых программных клиентов."""

    async def generate(self, prompt: str) -> str:
        return await self._client.generate(self._settings.ollama_chat_model, prompt)
