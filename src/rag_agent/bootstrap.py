from __future__ import annotations

from asyncio import Lock
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from rag_agent.core.config import Settings, get_settings
from rag_agent.domain.errors import AgentError
from rag_agent.integrations.ollama.client import OllamaClient
from rag_agent.services.directory_indexing_service import DirectoryIndexingService
from rag_agent.services.document_ingestion_service import DocumentIngestionService
from rag_agent.services.embedding_service import EmbeddingService
from rag_agent.services.retrieval_service import RetrievalService
from rag_agent.storage.document_store import DocumentStore
from rag_agent.storage.vector_store import JsonVectorStore


if TYPE_CHECKING:
    from rag_agent.agents.main_agent import RedTeamAgent


@dataclass
class ApplicationServices:
    settings: Settings
    retrieval: RetrievalService
    ingestion: DocumentIngestionService
    indexing: DirectoryIndexingService
    _stack: AsyncExitStack = field(repr=False)
    agent: RedTeamAgent | None = field(default=None, init=False)
    _agent_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    async def get_agent(self) -> RedTeamAgent:
        async with self._agent_lock:
            if self._closed:
                raise AgentError("Сервисы приложения уже закрыты.")
            if self.agent is None:
                # Поисковый режим даже не импортирует библиотеки chat-модели.
                from langchain_ollama import ChatOllama
                from rag_agent.agents.main_agent import RedTeamAgent

                # LangChain не предоставляет общего close для модели. Владеем
                # транспортами явно, чтобы закрыть пулы без приватных атрибутов.
                # trust_env=False исключает дополнительные неуправляемые proxy-пулы.
                async with AsyncExitStack() as resources:
                    sync_transport = resources.enter_context(httpx.HTTPTransport())
                    async_transport = await resources.enter_async_context(httpx.AsyncHTTPTransport())
                    llm = ChatOllama(
                        model=self.settings.ollama_chat_model,
                        base_url=str(self.settings.ollama_base_url),
                        temperature=0.3,
                        sync_client_kwargs={
                            "transport": sync_transport,
                            "timeout": self.settings.ollama_timeout,
                            "trust_env": False,
                        },
                        async_client_kwargs={
                            "transport": async_transport,
                            "timeout": self.settings.ollama_timeout,
                            "trust_env": False,
                        },
                    )
                    agent = RedTeamAgent(self.settings, llm)
                    await agent.initialize(retrieval_service=self.retrieval)
                    self._stack.push_async_callback(resources.pop_all().aclose)
                    self.agent = agent
            return self.agent


@asynccontextmanager
async def open_services(settings: Settings | None = None):
    """Открыть сервисы для CLI, HTTP или будущего MCP; закрыть при любом выходе."""
    settings = settings if settings is not None else get_settings()
    async with AsyncExitStack() as stack:
        client = OllamaClient(str(settings.ollama_base_url), settings.ollama_timeout)
        stack.push_async_callback(client.close)
        embeddings = EmbeddingService(client, settings)
        vector_store = JsonVectorStore(settings.index_path)
        ingestion = DocumentIngestionService(
            settings, DocumentStore(settings.documents_dir), vector_store, embeddings
        )
        services = ApplicationServices(
            settings=settings,
            retrieval=RetrievalService(settings, vector_store, embeddings),
            ingestion=ingestion,
            indexing=DirectoryIndexingService(settings, vector_store, ingestion),
            _stack=stack,
        )
        try:
            yield services
        finally:
            services._closed = True
