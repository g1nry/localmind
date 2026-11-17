from langchain_core.tools import StructuredTool, create_schema_from_function

from .base import BaseTool


class LangChainToolAdapter(StructuredTool):
    """Передать LangGraph реальные имена и типы аргументов инструмента."""

    def __init__(self, tool: BaseTool):
        super().__init__(
            name=tool.metadata.name,
            description=tool.metadata.description,
            args_schema=create_schema_from_function(
                tool.metadata.name, tool.arun, filter_args=["kwargs"]
            ),
            coroutine=tool.arun,
        )
