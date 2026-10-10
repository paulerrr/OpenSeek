#!/usr/bin/env python3
"""
OpenSeek server — OpenAI-compatible /v1/chat/completions backed by DeepSeek browser automation.

Usage:
    pip install fastapi uvicorn
    DEEPSEEK_EMAIL=you@example.com DEEPSEEK_PASSWORD=secret python server.py

Then point openclaude at http://localhost:8888/v1:
    cd ~/projects/openclaude
    bun run profile:init -- --provider openai --model deepseek-chat \
        --base-url http://localhost:8888/v1 --api-key dummy
    bun run dev
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from openseek import DeepSeek

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("OPENSEEK_PORT", "8888"))
EMAIL = os.getenv("DEEPSEEK_EMAIL", "")
PASSWORD = os.getenv("DEEPSEEK_PASSWORD", "")


class _State:
    deepseek: DeepSeek | None = None
    last_first_user_msg: str | None = None  # tracks conversation identity
    had_first_request: bool = False


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not EMAIL or not PASSWORD:
        logger.error("DEEPSEEK_EMAIL and DEEPSEEK_PASSWORD must be set.")
        sys.exit(1)
    logger.info("Launching DeepSeek browser session...")
    state.deepseek = DeepSeek(email=EMAIL, password=PASSWORD, headless=True)
    await state.deepseek.initialize()
    logger.info("Ready — listening on http://localhost:%d/v1", PORT)
    yield
    logger.info("Shutting down...")
    await state.deepseek.close()


app = FastAPI(lifespan=lifespan)


class _Message(BaseModel):
    role: str
    content: str | list


class _ChatRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: list[_Message]
    stream: bool = False
    max_tokens: int = 8192
    temperature: float = 1.0


def _to_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
        for b in content
    ).strip()


async def _start_new_chat(ds: DeepSeek) -> None:
    """Navigate to a fresh chat instead of clicking the button (more reliable in headless)."""
    await ds.browser.main_tab.get("https://chat.deepseek.com/")
    await ds.browser.main_tab.wait_for(ds.selectors.interactions.textbox, timeout=15)


@app.post("/v1/chat/completions")
async def chat_completions(req: _ChatRequest):
    ds = state.deepseek
    if ds is None:
        raise HTTPException(status_code=503, detail="Session not ready")

    logger.info("Request: %d messages, roles: %s", len(req.messages), [m.role for m in req.messages])

    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message")

    first_user = _to_text(user_msgs[0].content)
    last_user = _to_text(user_msgs[-1].content)

    logger.info("First user msg: %s...", first_user[:60])
    logger.info("Last user msg: %s...", last_user[:60])

    # Detect a new conversation (different first message from a prior session)
    is_new = state.last_first_user_msg != first_user
    state.last_first_user_msg = first_user

    # After login the browser is already on a fresh chat — only reset on new conversations
    if is_new and state.had_first_request:
        logger.info("New conversation — navigating to fresh chat")
        await _start_new_chat(ds)
    state.had_first_request = True

    logger.info("→ %s...", last_user[:80])
    response = await ds.send_message(last_user, timeout=120)
    text = response.text if response else ""
    logger.info("← %s...", text[:80])

    return {
        "id": f"chatcmpl-openseek-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "deepseek-chat", "object": "model", "created": 0, "owned_by": "openseek"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "session_active": state.deepseek is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
