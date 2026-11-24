from fastapi import FastAPI, Request, APIRouter
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse

from .bootstrap import open_services
from .core.config import Settings
from .domain.errors import AgentError, ProviderError


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    use_rag: bool = True
    top_k: int = 4


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with open_services(settings) as services:
            app.state.services = services
            try:
                yield
            finally:
                del app.state.services

    application = FastAPI(
        title="LocalMind",
        description="Local LLM agent for personal knowledge bases",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.exception_handler(AgentError)
    @application.exception_handler(ProviderError)
    async def service_error(request: Request, exc: Exception):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    application.include_router(router)
    return application


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>LocalMind</title></head>
        <body style="font-family: sans-serif; padding: 40px;">
            <h1>🤖 LocalMind</h1>
            <p>Backend is running.</p>
            <p>Main endpoints:</p>
            <ul>
                <li><code>POST /api/v1/chat</code></li>
                <li><code>POST /api/agent/chat</code></li>
                <li><code>GET /health</code></li>
            </ul>
        </body>
    </html>
    """


@router.post("/api/agent/chat")
async def agent_chat(payload: ChatRequest, request: Request):
    agent = await request.app.state.services.get_agent()
    result = await agent.ainvoke(
        message=payload.message,
        thread_id=payload.thread_id
    )
    return {
        "answer": result.get("response", ""),
        "contexts": []
    }


@router.post("/api/v1/chat")
async def legacy_chat(payload: ChatRequest, request: Request):
    return await agent_chat(payload, request)


@router.get("/health")
async def health(request: Request):
    agent = request.app.state.services.agent
    return {"status": "ok", "agent_initialized": agent is not None and agent.initialized}


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
