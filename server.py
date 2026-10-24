#!/usr/bin/env python3
"""
OpenSeek server — OpenAI-compatible /v1/chat/completions backed by DeepSeek browser automation.

Supports system prompts, streaming (with DeepThink output as `reasoning_content`), tool calling
(`tools` / `tool_calls`, implemented through a prompt protocol), attachments (`image_url` and `file` content parts)
and deep research (a plan, search steps and a report, run as several DeepSeek turns).

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
import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
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
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, os.path.dirname(__file__))
from openseek import DeepSeek
from openseek.internal.exceptions import FileUploadError, OpenSeekException, RateLimited
from openseek.internal.prompt import DEFAULT_SYSTEM_PROMPT
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.research import stream_research
from openseek.internal.tools import (
    CUT_NOTE,
    FabricatedResultGuard,
    ToolCallStreamFilter,
    ToolOutputGuard,
    build_tools_prompt,
    format_tool_call,
    format_tool_result,
    parse_tool_calls,
    tool_turn_instructions,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("OPENSEEK_PORT", "8888"))
EMAIL = os.getenv("DEEPSEEK_EMAIL", "")
PASSWORD = os.getenv("DEEPSEEK_PASSWORD", "")
TOKEN = os.getenv("DEEPSEEK_TOKEN", "")
USER_DATA_DIR = os.getenv("OPENSEEK_USER_DATA_DIR") or None
TIMEOUT = int(os.getenv("OPENSEEK_TIMEOUT", "300"))
REQUEST_TIMEOUT = float(os.getenv("OPENSEEK_REQUEST_TIMEOUT", "120"))
# A file whose contents replace the default instructions; an empty file disables them
SYSTEM_PROMPT_FILE = os.getenv("OPENSEEK_SYSTEM_PROMPT_FILE") or None
# Answer opencode's session-title requests locally instead of tying up the browser; "0" sends them to DeepSeek
LOCAL_TITLES = os.getenv("OPENSEEK_LOCAL_TITLES", "1") != "0"
# When set, /v1 requests must send it as a Bearer token (the OpenAI SDK's api_key); unset leaves the API open
API_KEY = os.getenv("OPENSEEK_API_KEY") or None
# Where opencode sessions and their DeepSeek chats are saved, so they survive a restart
STATE_FILE = os.getenv("OPENSEEK_STATE_FILE") or (
    str(Path(USER_DATA_DIR) / "openseek-sessions.json") if USER_DATA_DIR else None
)
MAX_CONVERSATIONS = 64
MAX_SESSIONS = 256
# DeepSeek's per-message attachment limits
MAX_FILES = 50
MAX_FILE_SIZE = 100 * 1024 * 1024

DATA_URL_RE = re.compile(r"^data:([^;,]*)((?:;[^;,]*)*),(.*)$", re.DOTALL)
MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}

SYSTEM_ROLES = ("system", "developer")
# Start of opencode's title agent prompt
TITLE_PROMPT_PREFIX = "You are a title generator"
# Starts of opencode's compaction prompts, for a first summary and for updating an earlier one
COMPACTION_PROMPT_PREFIXES = (
    "You MUST summarize the conversation above into a structured summary",
    "Update the existing checkpoint in the conversation above",
)
MAX_TITLE_LENGTH = 50
# Seconds between SSE comments sent while a request waits, so clients don't time out a queued request
KEEPALIVE_INTERVAL = 15


class _Chat(BaseModel):
    chat_id: str
    instructions_hash: str | None
    # Fingerprints of the messages the chat holds; kept only for sessions
    keys: list[str] = []
    # The last reply was cut at tool results the model invented, which stay in the chat; the next turn says so
    cut: bool = False


class _State:
    deepseek: DeepSeek | None = None
    # Fingerprint of every message in a conversation so far -> the DeepSeek chat holding it
    chats: OrderedDict[str, _Chat] = OrderedDict()
    # opencode session ID (the x-opencode-session-id header) -> the DeepSeek chat holding the session
    sessions: OrderedDict[str, _Chat] = OrderedDict()


state = _State()


def _load_sessions() -> None:
    if not STATE_FILE:
        return
    try:
        data = json.loads(Path(STATE_FILE).read_text())
        state.sessions = OrderedDict((key, _Chat(**value)) for key, value in data.items())
        logger.info("Loaded %d sessions from %s", len(state.sessions), STATE_FILE)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Could not load sessions from %s: %s", STATE_FILE, e)


def _save_sessions() -> None:
    if not STATE_FILE:
        return
    data = {key: chat.model_dump() for key, chat in state.sessions.items()}
    try:
        Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{STATE_FILE}.tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(STATE_FILE)
    except OSError as e:
        logger.warning("Could not save sessions to %s: %s", STATE_FILE, e)


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
    _load_sessions()
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


def _check_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    if API_KEY is None:
        return
    scheme, _, key = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(key.strip().encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key", headers={"WWW-Authenticate": "Bearer"})


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
    # Extensions: force DeepThink / Search / DeepResearch regardless of the model name
    deepthink: bool | None = None
    search: bool | None = None
    deepresearch: bool | None = None


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
        try:
            await stream.aclose()
        finally:
            if directory:
                shutil.rmtree(directory, ignore_errors=True)


async def _guard_tool_output(stream, forget=lambda: None):
    guard = ToolOutputGuard()
    try:
        async for event in stream:
            if event.type == "text":
                text = guard.feed(event.content)
                if text:
                    yield StreamEvent("text", text)
            else:
                if event.type == "done":
                    rest = guard.finish()
                    if rest:
                        yield StreamEvent("text", rest)
                yield event
    except RateLimited:
        # A refused request did not add model output to the chat. Keep its mapping.
        raise
    except OpenSeekException:
        forget()
        raise
    finally:
        await stream.aclose()


async def _cut_fabricated_results(stream, tool_names, chat_id=lambda: None, on_cut=lambda: None):
    """
    Ends a reply where the model, after its tool calls, starts writing the tool results itself.

    `chat_id` returns the chat being written to, read at the cut so the reply can still be remembered, and `on_cut`
    is called once the reply is cut.
    """
    guard = FabricatedResultGuard(tool_names)
    thinking = []
    cut_chat_id = None
    try:
        async for event in stream:
            if event.type == "text":
                text = guard.feed(event.content)
                if text:
                    yield StreamEvent("text", text)
                if guard.cut:
                    cut_chat_id = chat_id()
                    break
            else:
                if event.type == "thinking":
                    thinking.append(event.content)
                elif event.type == "done":
                    rest = guard.finish()
                    if rest:
                        yield StreamEvent("text", rest)
                yield event
    finally:
        # Closing the stream before it completes reloads the page, which stops DeepSeek's generation
        await stream.aclose()
    if guard.cut:
        logger.warning("Cut the reply at a tool result the model wrote itself")
        # The invented results stay in the website's chat. Replaying a long session into a new chat can take
        # minutes, so the chat is kept and the next turn tells the model its invented results were discarded.
        on_cut()
        yield StreamEvent(
            "done",
            response=Response(guard.text, chat_id=cut_chat_id, deepthink_content="".join(thinking) or None),
        )


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _message_key(role: str, content, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> str:
    """
    Fingerprint of a message, stable across the client echoing it back.

    Preserve whitespace and tool payloads so edits cannot silently reuse stale browser context.
    """
    calls = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            # Clients such as opencode parse and reserialize arguments between turns.
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        calls.append([call.get("id"), call.get("type", "function"), function.get("name"), arguments])
    return _hash([role, _to_text(content), calls, tool_call_id])


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


def _match_session(
    session_id: str, conversation: list[_Message], keys: list[str], history_length: int
) -> tuple[_Chat | None, list[_Message], bool]:
    """
    Finds the DeepSeek chat of an opencode session, and the messages it still needs.

    Returns the chat, the messages after the ones it holds and whether the request is a compaction. The chat is
    `None`, with the pending turn, when the conversation no longer extends it: after a revert, an edited message, or
    the summary that starts a compacted conversation. A compaction request repeats only the older part of the
    conversation, so the chat gets just its prompt.
    """
    pending = conversation[history_length:]
    known = state.sessions.get(session_id)
    if known is None:
        return None, pending, False
    compaction = next(
        (
            i
            for i in range(len(conversation) - 1, history_length - 1, -1)
            if conversation[i].role == "user"
            and _to_text(conversation[i].content).startswith(COMPACTION_PROMPT_PREFIXES)
        ),
        None,
    )
    if compaction is not None:
        return known, conversation[compaction:], True
    held = len(known.keys)
    # Every earlier reply must already be in the chat, so what's left is only new user and tool messages
    if history_length <= held < len(keys) and keys[:held] == known.keys:
        return known, conversation[held:], False
    return None, pending, False


def _is_title_request(system: str, history: list[_Message], tools: list[dict] | None) -> bool:
    return LOCAL_TITLES and not tools and not history and system.startswith(TITLE_PROMPT_PREFIX)


def _local_title(text: str) -> str:
    """A title from the first line of the user's message, cut at a word boundary."""
    line = next((line for line in text.splitlines() if line.strip()), "")
    title = " ".join(line.split()).strip(" .,:;!?")
    if len(title) > MAX_TITLE_LENGTH:
        title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0].rstrip(" .,:;!?")
    return title[:1].upper() + title[1:] if title else "New session"


def _static_completion(req: _ChatRequest, content: str):
    """A completion that doesn't go through DeepSeek."""
    completion_id = f"chatcmpl-openseek-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not req.stream:
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage,
        }

    def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            **extra,
        }
        return f"data: {json.dumps(body)}\n\n"

    async def events():
        yield chunk({"role": "assistant", "content": content})
        yield chunk({}, "stop", usage=usage)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _wants_deepthink(req: _ChatRequest) -> bool:
    if req.deepthink is not None:
        return req.deepthink
    model = req.model.lower()
    return "reasoner" in model or "think" in model


def _wants_deepresearch(req: _ChatRequest) -> bool:
    if req.deepresearch is not None:
        return req.deepresearch
    return "research" in req.model.lower()


@app.post("/v1/chat/completions", dependencies=[Depends(_check_api_key)])
async def chat_completions(req: _ChatRequest, x_opencode_session_id: Annotated[str | None, Header()] = None):
    deadline = asyncio.get_running_loop().time() + REQUEST_TIMEOUT
    try:
        return await asyncio.wait_for(
            _chat_completions(req, x_opencode_session_id, deadline), REQUEST_TIMEOUT
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="OpenSeek request deadline exceeded") from exc


async def _chat_completions(req, x_opencode_session_id, deadline):
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

    research = _wants_deepresearch(req)
    # A research run answers with a report, so tool calling is off for it
    tools = req.tools if req.tools and req.tool_choice != "none" and not research else None
    instructions = "\n\n".join(
        part
        for part in (DEFAULT_PROMPT, system, build_tools_prompt(tools) if tools else "")
        if part
    )
    instructions_hash = _hash(instructions) if instructions else None
    # Bare JSON calls without tags are accepted only for these names
    # Tool name -> parameters schema; the parser checks DSML parameters against it
    request_tools = {
        function.get("name"): function.get("parameters")
        for function in (tool.get("function", tool) for tool in tools)
    } if tools else None

    if _is_title_request(system, history, tools):
        title = _local_title(_render_turn(pending, {}))
        logger.info("Title request answered locally: %s", title)
        return _static_completion(req, title)

    tool_names = {
        call.get("id"): call.get("function", {}).get("name")
        for m in history
        if m.role == "assistant"
        for call in m.tool_calls or []
    }
    keys = [_message_key(m.role, m.content, m.tool_calls, m.tool_call_id) for m in conversation]
    session_id = x_opencode_session_id
    # Sessions are matched by ID, other clients by the fingerprint of their history
    if session_id:
        known, new, compacting = _match_session(session_id, conversation, keys, len(history))
    else:
        known, new, compacting = state.chats.pop(_conversation_key(keys[: len(history)]), None), pending, False

    # A known chat already holds earlier attachments; a replayed conversation needs all of them again
    attachments = [a for m in (new if known else conversation) for a in _attachments(m.content)]
    if len(attachments) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_FILES} attachments are supported")
    directory = tempfile.mkdtemp(prefix="openseek-") if attachments else None
    try:
        files = await _save_attachments(attachments, directory) if attachments else None
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    if known:
        chat_id = known.chat_id
        message = _render_turn(new, tool_names)
        if instructions_hash != known.instructions_hash:
            system_prompt = instructions or None
        else:
            system_prompt = None
        if known.cut:
            system_prompt = "\n\n".join(part for part in (system_prompt, CUT_NOTE) if part)
        if compacting:
            logger.info("Summarizing session %s in chat %s", session_id, chat_id)
        else:
            logger.info("Continuing chat %s (%d new messages)", chat_id, len(new))
    else:
        chat_id = None
        turn = _render_turn(pending, tool_names)
        message = f"{_render_history(history, tool_names)}\n\n{turn}" if history else turn
        system_prompt = instructions or None
        logger.info("New chat (%d earlier messages replayed)", len(history))

    system_prompt = "\n\n".join(
        part for part in (system_prompt, tool_turn_instructions(tools, req.tool_choice)) if part
    )

    cut = False

    def on_cut() -> None:
        nonlocal cut
        cut = True

    def remember(response, content: str, calls: list[dict]) -> None:
        # The chat stays as it was before a summary; the compacted conversation after it starts a new chat
        if compacting or not response.chat_id:
            return
        reply_key = _message_key("assistant", content, calls)
        if session_id:
            state.sessions[session_id] = _Chat(
                chat_id=response.chat_id, instructions_hash=instructions_hash, keys=keys + [reply_key], cut=cut
            )
            state.sessions.move_to_end(session_id)
            while len(state.sessions) > MAX_SESSIONS:
                state.sessions.popitem(last=False)
            _save_sessions()
            return
        state.chats[_conversation_key(keys + [reply_key])] = _Chat(
            chat_id=response.chat_id, instructions_hash=instructions_hash, cut=cut
        )
        while len(state.chats) > MAX_CONVERSATIONS:
            state.chats.popitem(last=False)

    if files:
        logger.info("Attaching %d files", len(files))
    if research:
        logger.info("Starting deep research")
        stream = stream_research(
            ds,
            message,
            deepthink=_wants_deepthink(req),
            timeout=TIMEOUT,
            chat_id=chat_id,
            new_chat=known is None,
            system_prompt=system_prompt,
            files=files,
        )
    else:
        stream = ds.stream_message(
            message,
            deepthink=_wants_deepthink(req),
            search=bool(req.search),
            timeout=TIMEOUT,
            chat_id=chat_id,
            system_prompt=system_prompt,
            files=files,
            new_chat=known is None,
        )
    if tools:
        def forget():
            # Failed output may already be in the website's history. Replay on retry.
            if session_id:
                state.sessions.pop(session_id, None)
                _save_sessions()

        stream = _guard_tool_output(stream, forget)
        stream = _cut_fabricated_results(stream, request_tools, lambda: chat_id or ds.active_chat_id, on_cut)
    stream = _request_deadline(
        _remove_after(stream, directory), deadline=deadline
    )
    completion_id = f"chatcmpl-openseek-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def restore_rate_limited_chat():
        if known and not session_id:
            state.chats[_conversation_key(keys[: len(history)])] = known

    if req.stream:
        return StreamingResponse(
            _stream_completion(stream, completion_id, created, req.model, request_tools, remember,
                               restore_rate_limited_chat),
            media_type="text/event-stream",
        )

    response = None
    try:
        async for event in stream:
            if event.type == "done":
                response = event.response
    except FileUploadError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RateLimited as e:
        restore_rate_limited_chat()
        raise HTTPException(status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)}) from e
    except OpenSeekException as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if response is None:
        raise HTTPException(status_code=504, detail="Timed out waiting for DeepSeek")

    content, calls = parse_tool_calls(response.text, request_tools) if tools else (response.text, [])
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


async def _stream_completion(stream, completion_id: str, created: int, model: str, tool_names, remember,
                             on_rate_limit=lambda: None):
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

    def error(message: str, code: int, **extra) -> str:
        kind = "rate_limit_error" if code == 429 else "server_error"
        return f"data: {json.dumps({'error': {'message': message, 'type': kind, 'code': code, **extra}})}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    # None when the request has no tools; a (possibly empty) set of names otherwise
    text_filter = ToolCallStreamFilter(tool_names) if tool_names is not None else None
    emitted_text = []
    response = None
    try:
        async for event in _with_keepalive(stream):
            if event is None:
                yield ": keepalive\n\n"
            elif event.type == "thinking":
                yield chunk({"reasoning_content": event.content})
            elif event.type == "text":
                text = text_filter.feed(event.content) if text_filter else event.content
                if text:
                    emitted_text.append(text)
                    yield chunk({"content": text})
            elif event.type == "done":
                response = event.response
    except FileUploadError as e:
        logger.error("Stream failed: %s", e)
        yield error(str(e), 400)
        yield "data: [DONE]\n\n"
        return
    except RateLimited as e:
        on_rate_limit()
        logger.warning("Stream rate limited: %s", e)
        yield error(str(e), 429, retry_after=e.retry_after)
        yield "data: [DONE]\n\n"
        return
    except OpenSeekException as e:
        logger.error("Stream failed: %s", e)
        yield error(str(e), 503)
        yield "data: [DONE]\n\n"
        return

    if response is None:
        logger.error("Timed out waiting for DeepSeek")
        # An error, not finish_reason "length", so clients retry instead of treating the reply as truncated
        yield error("Timed out waiting for DeepSeek", 504)
        yield "data: [DONE]\n\n"
        return

    calls = []
    content = response.text
    if text_filter:
        rest = text_filter.finish()
        content, calls = parse_tool_calls(response.text, tool_names)
        if rest:
            emitted_text.append(rest)
            yield chunk({"content": rest})
        if calls:
            yield chunk(
                {"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]},
            )

    # Cache exactly what the streaming client receives, including significant whitespace.
    remember(response, "".join(emitted_text), calls)
    logger.info("← streamed %d chars, %d tool calls", len(content), len(calls))
    yield chunk({}, "tool_calls" if calls else "stop", usage=_usage(response))
    yield "data: [DONE]\n\n"


async def _request_deadline(stream, timeout=None, *, deadline=None):
    """One budget for queueing and all turns, including multi-turn research."""
    if deadline is None:
        deadline = asyncio.get_running_loop().time() + timeout
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    anext(stream), max(0, deadline - asyncio.get_running_loop().time())
                )
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                raise OpenSeekException("OpenSeek request deadline exceeded") from exc
            yield event
    finally:
        await stream.aclose()


async def _with_keepalive(stream):
    """Passes a stream through, yielding None whenever it has been silent for `KEEPALIVE_INTERVAL` seconds."""
    iterator = aiter(stream)
    pending = None
    try:
        while True:
            pending = pending or asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=KEEPALIVE_INTERVAL)
            if not done:
                yield None
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            finally:
                pending = None
            yield event
    finally:
        if pending:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        await iterator.aclose()


@app.get("/v1/models", dependencies=[Depends(_check_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "openseek"}
            for model in ("deepseek-chat", "deepseek-reasoner", "deepseek-research")
        ],
    }


@app.get("/health")
async def health():
    ds = state.deepseek
    ready = ds is not None and ds._initialized and ds._healthy
    if not ready:
        raise HTTPException(status_code=503, detail="DeepSeek browser unavailable or recovering")
    return {"status": "ok", "session_active": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
