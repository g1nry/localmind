from math import isfinite

import httpx

from rag_agent.domain.errors import ProviderError


class OllamaClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            response = await self.client.post(f"{self.base_url}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError("Ollama недоступна или вернула некорректный ответ.") from exc

    async def generate(self, model: str, prompt: str) -> str:
        result = await self._post(
            "/api/generate", {"model": model, "prompt": prompt, "stream": False}
        )
        text = result.get("response") if isinstance(result, dict) else None
        if not isinstance(text, str):
            raise ProviderError("Ollama не вернула текст ответа.")
        return text

    async def embed(self, model: str, text: str) -> list[float]:
        result = await self._post("/api/embeddings", {"model": model, "prompt": text})
        embedding = result.get("embedding") if isinstance(result, dict) else None
        if not isinstance(embedding, list) or not embedding or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
            for value in embedding
        ):
            raise ProviderError("Ollama не вернула корректный embedding.")
        return embedding

    async def close(self):
        await self.client.aclose()
