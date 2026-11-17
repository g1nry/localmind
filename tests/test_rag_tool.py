import asyncio

from rag_agent.domain.schemas import ContextItem, RetrievalResult
from rag_agent.tools.langchain_adapter import LangChainToolAdapter
from rag_agent.tools.rag_tool import RAGTool
from rag_agent.tools.registry import ToolRegistry


class FakeRetrieval:
    async def retrieve_context(self, query):
        assert query == "question"
        return RetrievalResult(
            contexts=[ContextItem(text="Known fact", source="notes.md", chunk_id="notes:0")]
        )


def test_agent_tool_returns_typed_retrieval_results():
    tool = LangChainToolAdapter(RAGTool(FakeRetrieval()))
    result = asyncio.run(tool.ainvoke({"query": "question"}))
    assert "Known fact" in result
    assert "notes.md" in result
    assert set(tool.args) == {"query"}


def test_default_registry_only_exposes_knowledge_search():
    registry = ToolRegistry()
    asyncio.run(registry.initialize(FakeRetrieval()))
    assert list(registry.tools) == ["rag_search"]
