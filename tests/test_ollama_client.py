import asyncio

import httpx
import pytest

from rag_agent.domain.errors import ProviderError
from rag_agent.integrations.ollama.client import OllamaClient


@pytest.mark.parametrize('body', [b'{}', b'{"embedding": []}', b'{"embedding": [true]}',
                                  b'{"embedding": [NaN]}', b'not-json'])
def test_bad_embedding_response_is_explicit_error(body):
    async def run():
        client = OllamaClient('http://ollama', 1)
        await client.close()
        client.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        )
        try:
            with pytest.raises(ProviderError):
                await client.embed('embedding-model', 'question')
        finally:
            await client.close()
    asyncio.run(run())
