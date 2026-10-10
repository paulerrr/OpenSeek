#!/usr/bin/env python3
"""
OpenSeek server — OpenAI-compatible /v1/chat/completions backed by DeepSeek browser automation.

Supports system prompts, streaming (with DeepThink output as `reasoning_content`), tool calling
(`tools` / `tool_calls`, implemented through a prompt protocol) and attachments (`image_url` and `file` content parts).

Usage:
    pip install fastapi uvicorn
    DEEPSEEK_TOKEN=<userToken> python server.py
    # or: DEEPSEEK_EMAIL=you@example.com DEEPSEEK_PASSWORD=secret python server.py

Then point openclaude at http://localhost:8888/v1:
    cd ~/projects/openclaude
    bun run profile:init -- --provider openai --model deepseek-chat \
        --base-url http://localhost:8888/v1 --api-key dummy
    bun run dev
"""

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, os.path.dirname(__file__))
from openseek import DeepSeek
from openseek.internal.exceptions import FileUploadError, OpenSeekException
from openseek.internal.prompt import DEFAULT_SYSTEM_PROMPT
from openseek.internal.tools import (
    BLOCK_RE,
    TOOLS_REMINDER,
    ToolCallStreamFilter,
    build_tools_prompt,
    format_tool_call,
    format_tool_result,
    parse_tool_calls,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("OPENSEEK_PORT", "8888"))
EMAIL = os.getenv("DEEPSEEK_EMAIL", "")
PASSWORD = os.getenv("DEEPSEEK_PASSWORD", "")
TOKEN = os.getenv("DEEPSEEK_TOKEN", "")
USER_DATA_DIR = os.getenv("OPENSEEK_USER_DATA_DIR") or None
TIMEOUT = int(os.getenv("OPENSEEK_TIMEOUT", "300"))
# A file whose contents replace the default instructions; an empty file disables them
SYSTEM_PROMPT_FILE = os.getenv("OPENSEEK_SYSTEM_PROMPT_FILE") or None
MAX_CONVERSATIONS = 64
# DeepSeek's per-message attachment limits
MAX_FILES = 50
MAX_FILE_SIZE = 100 * 1024 * 1024

DATA_URL_RE = re.compile(r"^data:([^;,]*)((?:;[^;,]*)*),(.*)$", re.DOTALL)
MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}

SYSTEM_ROLES = ("system", "developer")


class _Chat(BaseModel):
    chat_id: str
    instructions_hash: str | None


class _State:
    deepseek: DeepSeek | None = None
    # Fingerprint of every message in a conversation so far -> the DeepSeek chat holding it
    chats: OrderedDict[str, _Chat] = OrderedDict()


state = _State()


def _load_default_prompt() -> str:
    if not SYSTEM_PROMPT_FILE:
        return DEFAULT_SYSTEM_PROMPT
    try:
        return Path(SYSTEM_PROMPT_FILE).read_text().strip()
    except OSError as e:
        logger.error("Could not read OPENSEEK_SYSTEM_PROMPT_FILE: %s", e)
        sys.exit(1)


DEFAULT_PROMPT = _load_default_prompt()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not TOKEN and not (EMAIL and PASSWORD):
        logger.error("DEEPSEEK_TOKEN, or DEEPSEEK_EMAIL and DEEPSEEK_PASSWORD, must be set.")
        sys.exit(1)
    logger.info("Launching DeepSeek browser session...")
    state.deepseek = DeepSeek(
        email=EMAIL, password=PASSWORD, token=TOKEN, headless=True, user_data_dir=USER_DATA_DIR
    )
    await state.deepseek.initialize()
    logger.info("Ready — listening on http://localhost:%d/v1", PORT)
    yield
    logger.info("Shutting down...")
    await state.deepseek.close()


app = FastAPI(lifespan=lifespan)


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class _ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "deepseek-chat"
    messages: list[_Message]
    stream: bool = False
    max_tokens: int = 8192
    temperature: float = 1.0
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    # Extensions: force DeepThink / Search regardless of the model name
    deepthink: bool | None = None
    search: bool | None = None


class _Attachment(BaseModel):
    name: str  # shown in the prompt; becomes the file name, with an extension added if it has none
    source: str  # data URL, http(s) URL or bare base64


def _attachment(block: dict) -> _Attachment | None:
    """Reads an `image_url` or `file` content part. Names are derived from the content, so they stay stable."""
    if block.get("type") == "image_url":
        image = block.get("image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if isinstance(url, str) and url:
            return _Attachment(name=f"image-{_hash(url)[:8]}", source=url)
    elif block.get("type") == "file":
        file = block.get("file") or {}
        data = file.get("file_data")
        if isinstance(data, str) and data:
            name = Path(file.get("filename") or "").name or f"file-{_hash(data)[:8]}"
            return _Attachment(name=name, source=data)
    return None


def _attachments(content: str | list | None) -> list[_Attachment]:
    if not isinstance(content, list):
        return []
    return [attachment for block in content if isinstance(block, dict) and (attachment := _attachment(block))]


def _to_text(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, dict) and (attachment := _attachment(block)):
            parts.append(f"[attached: {attachment.name}]")
        elif isinstance(block, dict):
            parts.append(f"[{block.get('type', 'content')} omitted]")
    return "\n".join(parts).strip()


def _download(url: str) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenSeek"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(MAX_FILE_SIZE + 1), response.headers.get_content_type()


async def _load(source: str) -> tuple[bytes, str | None]:
    """Returns an attachment's bytes and MIME type (if known)."""
    if match := DATA_URL_RE.match(source):
        mime, params, payload = match.groups()
        if ";base64" in params:
            data = base64.b64decode("".join(payload.split()), validate=True)
        else:
            data = urllib.parse.unquote_to_bytes(payload)
        return data, mime or None
    if source.startswith(("http://", "https://")):
        return await asyncio.to_thread(_download, source)
    return base64.b64decode("".join(source.split()), validate=True), None


async def _save_attachments(attachments: list[_Attachment], directory: str) -> list[str]:
    """Writes attachments into `directory`, returning their paths. Identical attachments are saved once."""
    paths = []
    for attachment in attachments:
        try:
            data, mime = await _load(attachment.source)
        except (ValueError, binascii.Error, OSError) as e:
            raise HTTPException(status_code=400, detail=f"Could not read attachment {attachment.name}: {e}") from e
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"Attachment {attachment.name} is larger than 100 MB")

        name = attachment.name
        if not Path(name).suffix:
            name += MIME_EXTENSIONS.get(mime or "") or mimetypes.guess_extension(mime or "") or ".bin"
        path = Path(directory) / name
        if path.exists():
            if path.read_bytes() == data:
                continue
            path = Path(directory) / f"{len(paths)}-{name}"
        path.write_bytes(data)
        paths.append(str(path))
    return paths


async def _remove_after(stream, directory: str | None):
    """Passes a stream through, deleting the attachments directory once it ends."""
    try:
        async for event in stream:
            yield event
    finally:
        if directory:
            shutil.rmtree(directory, ignore_errors=True)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _message_key(role: str, content, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> str:
    """Fingerprint of a message, stable across the client echoing it back."""
    call_ids = [call.get("id") for call in tool_calls or []]
    return _hash([role, _to_text(content).strip(), call_ids, tool_call_id])


def _conversation_key(keys: list[str]) -> str:
    return _hash(keys)


def _render_turn(messages: list[_Message], tool_names: dict[str, str]) -> str:
    """Renders the new user / tool messages of a turn as the text sent to DeepSeek."""
    parts = []
    for message in messages:
        if message.role == "tool":
            name = message.name or tool_names.get(message.tool_call_id or "")
            parts.append(format_tool_result(_to_text(message.content), message.tool_call_id, name))
        else:
            parts.append(_to_text(message.content))
    return "\n\n".join(part for part in parts if part)


def _render_history(messages: list[_Message], tool_names: dict[str, str]) -> str:
    """Renders earlier messages, for replaying a conversation into a fresh DeepSeek chat."""
    entries = []
    for message in messages:
        if message.role == "assistant":
            body = "\n".join(
                [_to_text(message.content)] + [format_tool_call(call) for call in message.tool_calls or []]
            ).strip()
            entries.append(f"<assistant>\n{body}\n</assistant>")
        elif message.role == "tool":
            entries.append(_render_turn([message], tool_names))
        else:
            entries.append(f"<user>\n{_to_text(message.content)}\n</user>")
    return (
        "Earlier messages of this conversation are below for context. Reply to the latest message after them.\n\n"
        "<conversation_history>\n" + "\n\n".join(entries) + "\n</conversation_history>"
    )


def _wants_deepthink(req: _ChatRequest) -> bool:
    if req.deepthink is not None:
        return req.deepthink
    model = req.model.lower()
    return "reasoner" in model or "think" in model


@app.post("/v1/chat/completions")
async def chat_completions(req: _ChatRequest):
    ds = state.deepseek
    if ds is None:
        raise HTTPException(status_code=503, detail="Session not ready")

    system = "\n\n".join(_to_text(m.content) for m in req.messages if m.role in SYSTEM_ROLES).strip()
    conversation = [m for m in req.messages if m.role not in SYSTEM_ROLES]

    # The turn to send is everything after the last assistant message
    last_assistant = max((i for i, m in enumerate(conversation) if m.role == "assistant"), default=-1)
    history, pending = conversation[: last_assistant + 1], conversation[last_assistant + 1 :]
    if not pending:
        raise HTTPException(status_code=400, detail="The last message must be a user or tool message")

    tools = req.tools if req.tools and req.tool_choice != "none" else None
    instructions = "\n\n".join(
        part
        for part in (DEFAULT_PROMPT, system, build_tools_prompt(tools, req.tool_choice) if tools else "")
        if part
    )
    instructions_hash = _hash(instructions) if instructions else None

    tool_names = {
        call.get("id"): call.get("function", {}).get("name")
        for m in history
        if m.role == "assistant"
        for call in m.tool_calls or []
    }
    keys = [_message_key(m.role, m.content, m.tool_calls, m.tool_call_id) for m in conversation]
    turn = _render_turn(pending, tool_names)

    history_key = _conversation_key(keys[: len(history)])
    known = state.chats.get(history_key)
    # A known chat already holds earlier attachments; a replayed conversation needs all of them again
    attachments = [a for m in (pending if known else conversation) for a in _attachments(m.content)]
    if len(attachments) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_FILES} attachments are supported")
    directory = tempfile.mkdtemp(prefix="openseek-") if attachments else None
    try:
        files = await _save_attachments(attachments, directory) if attachments else None
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    if known:
        state.chats.pop(history_key, None)
        chat_id = known.chat_id
        message = turn
        if instructions_hash != known.instructions_hash:
            system_prompt = instructions or None
        else:
            system_prompt = TOOLS_REMINDER if tools else None
        logger.info("Continuing chat %s (%d new messages)", chat_id, len(pending))
    else:
        await ds.start_new_chat()
        chat_id = None
        message = f"{_render_history(history, tool_names)}\n\n{turn}" if history else turn
        system_prompt = instructions or None
        logger.info("New chat (%d earlier messages replayed)", len(history))

    def remember(response, content: str, calls: list[dict]) -> None:
        reply_key = _message_key("assistant", content, calls)
        state.chats[_conversation_key(keys + [reply_key])] = _Chat(
            chat_id=response.chat_id, instructions_hash=instructions_hash
        )
        while len(state.chats) > MAX_CONVERSATIONS:
            state.chats.popitem(last=False)

    if files:
        logger.info("Attaching %d files", len(files))
    stream = _remove_after(
        ds.stream_message(
            message,
            deepthink=_wants_deepthink(req),
            search=bool(req.search),
            timeout=TIMEOUT,
            chat_id=chat_id,
            system_prompt=system_prompt,
            files=files,
        ),
        directory,
    )
    completion_id = f"chatcmpl-openseek-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream_completion(stream, completion_id, created, req.model, bool(tools), remember),
            media_type="text/event-stream",
        )

    response = None
    try:
        async for event in stream:
            if event.type == "done":
                response = event.response
    except FileUploadError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OpenSeekException as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if response is None:
        raise HTTPException(status_code=504, detail="Timed out waiting for DeepSeek")

    content, calls = parse_tool_calls(response.text) if tools else (response.text, [])
    remember(response, content, calls)
    logger.info("← %d chars, %d tool calls", len(content), len(calls))

    message_out = {"role": "assistant", "content": content or (None if calls else "")}
    if response.deepthink_content:
        message_out["reasoning_content"] = response.deepthink_content
    if calls:
        message_out["tool_calls"] = calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "message": message_out, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": _usage(response),
    }


def _usage(response) -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": response.token_usage or 0}


async def _stream_completion(stream, completion_id: str, created: int, model: str, tools: bool, remember):
    def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(body)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    text_filter = ToolCallStreamFilter() if tools else None
    response = None
    try:
        async for event in stream:
            if event.type == "thinking":
                yield chunk({"reasoning_content": event.content})
            elif event.type == "text":
                text = text_filter.feed(event.content) if text_filter else event.content
                if text:
                    yield chunk({"content": text})
            elif event.type == "done":
                response = event.response
    except OpenSeekException as e:
        logger.error("Stream failed: %s", e)
        yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    if response is None:
        logger.error("Timed out waiting for DeepSeek")
        yield chunk({}, "length")
        yield "data: [DONE]\n\n"
        return

    calls = []
    content = response.text
    if text_filter:
        rest = text_filter.finish()
        content, calls = parse_tool_calls(response.text)
        # Blocks that weren't valid calls were held back while streaming; send them as text now
        held_back = "".join(match.group(0) for match in BLOCK_RE.finditer(content)) + rest
        if held_back:
            yield chunk({"content": held_back})
        if calls:
            yield chunk(
                {"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]},
            )

    remember(response, content, calls)
    logger.info("← streamed %d chars, %d tool calls", len(content), len(calls))
    yield chunk({}, "tool_calls" if calls else "stop", usage=_usage(response))
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "openseek"}
            for model in ("deepseek-chat", "deepseek-reasoner")
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "session_active": state.deepseek is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
