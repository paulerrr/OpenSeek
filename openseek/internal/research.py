"""
Deep research, run as a sequence of DeepSeek turns in one chat.

DeepSeek's web chat has no research mode of its own, so this follows the plan-then-execute loop of browser extensions
such as Better DeepSeek: one turn writes a research plan, each step then runs as its own turn (a DeepSeek web search,
or a page OpenSeek downloads and attaches), and a last turn writes the report from the findings the chat now holds.
"""

import asyncio
import contextlib
import json
import logging
import re
import shutil
import tempfile
import time
import urllib.request
from collections.abc import AsyncIterator
from html.parser import HTMLParser
from pathlib import Path

from .exceptions import OpenSeekException, ServerDown
from .objects import Response, StreamEvent

logger = logging.getLogger(__name__)

MAX_STEPS = 6
# Seconds to wait before each retry of a turn DeepSeek rejected for coming too soon after the previous one
RATE_LIMIT_DELAYS = (15, 30, 60)
# Bytes read from a fetched page, and characters of its text that are attached
MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_PAGE_CHARS = 200_000

PLAN_RE = re.compile(r"<research_plan>(.*?)</research_plan>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

PLAN_PROMPT = """Deep research mode. Do not answer the request and do not search yet. Write only a research plan for \
the request below, as JSON inside <research_plan></research_plan> tags, and nothing else:
<research_plan>
{{"title": "short title", "steps": [{{"action": "search", "query": "...", "purpose": "..."}}]}}
</research_plan>

- Use 3 to {max_steps} steps. Each one is run separately, and you will see its results before the next.
- "action" is "search" for a web search, with a narrow query that names the entities, versions, dates or places that \
matter, or "fetch" to read one specific page, with its full URL as the query.
- "purpose" says what the step should establish.

Current date: {date}

Research request:
{request}"""

SEARCH_STEP_PROMPT = """Research step {number} of {total}: {purpose}
Search the web for: {query}

Report what you found that bears on the research request: concrete facts, figures and dates, each with its source \
URL. Note where sources disagree. Write in the language of the research request, whatever the language of the \
sources. Do not write the final report yet."""

FETCH_STEP_PROMPT = """Research step {number} of {total}: {purpose}
The attached file is the text of {url}

Report what it says that bears on the research request: concrete facts, figures and dates. Write in the language \
of the research request. Do not write the final report yet."""

REPORT_PROMPT = """All research steps are done.{skipped} Now write the final report that answers the research \
request, using the findings above, in the language of the request. Open with a short summary of the answer, then \
give the details under headings. Cite sources inline as Markdown links, and point out where sources disagree or \
information is missing."""


def parse_plan(text: str, max_steps: int = MAX_STEPS) -> tuple[str, list[dict]]:
    """
    Reads the plan from the planning turn's reply, returning its title and steps.

    Each step is a dict with "action" ("search" or "fetch"), "query" and "purpose". A fetch step without an http(s)
    URL becomes a search. Returns an empty step list when no usable plan is found.
    """
    data = None
    candidates = [m.group(1) for m in PLAN_RE.finditer(text)] + [m.group(1) for m in FENCE_RE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        if start == -1:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            data, _ = json.JSONDecoder().raw_decode(candidate[start:])
            if isinstance(data, dict):
                break
            data = None
    if data is None:
        return "", []

    steps = []
    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue
        query = str(step.get("query") or "").strip()
        if not query:
            continue
        action = str(step.get("action") or "search").strip().lower()
        if action != "fetch" or not query.startswith(("http://", "https://")):
            action = "search"
        steps.append({"action": action, "query": query, "purpose": str(step.get("purpose") or "").strip()})
    return str(data.get("title") or "").strip(), steps[:max_steps]


class _TextExtractor(HTMLParser):
    """Collects the readable text of an HTML page, without scripts, styles and other non-content elements."""

    SKIPPED = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCKS = {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIPPED:
            self._skipping += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIPPED:
            self._skipping = max(0, self._skipping - 1)
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skipping:
            self.parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _fetch_page(url: str) -> str:
    """Downloads a page and returns its text."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; OpenSeek)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(MAX_PAGE_BYTES)
        charset = response.headers.get_content_charset() or "utf-8"
        content_type = response.headers.get_content_type()
    text = body.decode(charset, errors="replace")
    if "html" in content_type or text.lstrip()[:1] == "<":
        extractor = _TextExtractor()
        extractor.feed(text)
        text = extractor.text()
    return text[:MAX_PAGE_CHARS]


def _describe(step: dict) -> str:
    return f"{step['action']}: {step['query']}" + (f" ({step['purpose']})" if step["purpose"] else "")


async def stream_research(
    ds,
    request: str,
    *,
    deepthink: bool = False,
    timeout: int = 300,
    chat_id: str | None = None,
    new_chat: bool = False,
    system_prompt: str | None = None,
    files: list[str] | None = None,
    max_steps: int = MAX_STEPS,
) -> AsyncIterator[StreamEvent]:
    """
    Researches `request` over several turns of one DeepSeek chat, yielding events like `DeepSeek.stream_message`.

    The plan, each step's findings and DeepThink output arrive as "thinking" events, the final report as "text"
    events, then a "done" event whose `Response` holds the report, with the research log as its `deepthink_content`.
    `chat_id`, `new_chat`, `system_prompt` and `files` apply to the planning turn, which starts or continues the chat;
    the following turns stay in the chat it ends up in.

    A step that fails or times out is skipped. Stops without a "done" event if the planning or report turn times out,
    and raises if either of them fails.
    """
    log: list[str] = []
    search_results: list = []

    def progress(text: str) -> StreamEvent:
        log.append(text)
        return StreamEvent("thinking", text)

    async def turn(message: str, text: str = "progress", **kwargs) -> AsyncIterator[StreamEvent]:
        """
        One turn. Its reasoning streams as research progress, and its text as progress, as the answer ("answer") or
        not at all ("hide"). Retries when DeepSeek rejects the message for coming too soon, unless answer text has
        already been streamed.
        """
        for delay in (*RATE_LIMIT_DELAYS, None):
            answered = False
            try:
                async for event in ds.stream_message(message, deepthink=deepthink, timeout=timeout, **kwargs):
                    if event.type == "thinking" or (event.type == "text" and text == "progress"):
                        yield progress(event.content)
                    elif event.type == "text" and text == "answer":
                        answered = True
                        yield event
                    elif event.type == "done":
                        yield event
                return
            except ServerDown as e:
                if delay is None or answered or "too frequent" not in str(e).lower():
                    raise
                logger.info("DeepSeek rate-limited the research turn, retrying in %ds", delay)
                yield progress(f"\nDeepSeek says messages are too frequent; retrying in {delay}s.\n")
                await asyncio.sleep(delay)

    yield progress("Planning the research...\n\n")
    plan = None
    plan_prompt = PLAN_PROMPT.format(max_steps=max_steps, date=time.strftime("%Y-%m-%d"), request=request)
    async for event in turn(
        plan_prompt, text="hide", chat_id=chat_id, new_chat=new_chat, system_prompt=system_prompt, files=files
    ):
        if event.type == "done":
            plan = event.response
        else:
            yield event
    if plan is None:
        return
    chat_id = plan.chat_id

    title, steps = parse_plan(plan.text, max_steps)
    if not steps:
        logger.warning("No usable research plan in the reply; researching the request directly")
        steps = [{"action": "search", "query": request[:500], "purpose": "Answer the research request"}]
    yield progress(
        f"\n\n**Research plan{': ' + title if title else ''}**\n"
        + "".join(f"{i}. {_describe(step)}\n" for i, step in enumerate(steps, 1))
    )

    skipped = []
    directory = tempfile.mkdtemp(prefix="openseek-research-")
    try:
        for number, step in enumerate(steps, 1):
            yield progress(f"\n\n**Step {number}/{len(steps)}** {_describe(step)}\n\n")
            kwargs = {"chat_id": chat_id}
            if step["action"] == "fetch":
                try:
                    text = await asyncio.to_thread(_fetch_page, step["query"])
                except (OSError, ValueError) as e:
                    logger.warning("Could not fetch %s: %s", step["query"], e)
                    yield progress(f"Could not fetch the page ({e}).\n")
                    skipped.append(number)
                    continue
                path = Path(directory) / f"step-{number}.txt"
                path.write_text(f"Source: {step['query']}\n\n{text}")
                prompt = FETCH_STEP_PROMPT.format(
                    number=number, total=len(steps), purpose=step["purpose"], url=step["query"]
                )
                kwargs["files"] = [str(path)]
            else:
                prompt = SEARCH_STEP_PROMPT.format(
                    number=number, total=len(steps), purpose=step["purpose"], query=step["query"]
                )
                # DeepSeek doesn't allow search on a message with attachments, so only search steps use it
                kwargs["search"] = True

            response = None
            try:
                async for event in turn(prompt, **kwargs):
                    if event.type == "done":
                        response = event.response
                    else:
                        yield event
            except OpenSeekException as e:
                logger.warning("Research step %d failed: %s", number, e)
                yield progress(f"\nThe step failed ({e}).\n")
            if response is None:
                skipped.append(number)
                continue
            chat_id = response.chat_id or chat_id
            search_results.extend(response.search_results or [])
            logger.info(
                "Research step %d/%d done: %d chars, %d search results",
                number,
                len(steps),
                len(response.text),
                len(response.search_results or []),
            )
    finally:
        shutil.rmtree(directory, ignore_errors=True)

    yield progress("\n\n**Writing the report...**\n")
    note = f" Steps {', '.join(map(str, skipped))} failed, so their findings are missing." if skipped else ""
    async for event in turn(REPORT_PROMPT.format(skipped=note), text="answer", chat_id=chat_id):
        if event.type == "done":
            response = event.response
            yield StreamEvent(
                "done",
                response=Response(
                    text=response.text,
                    chat_id=response.chat_id or chat_id,
                    deepthink_content="".join(log).strip() or None,
                    search_results=(search_results + (response.search_results or [])) or None,
                    message_id=response.message_id,
                    token_usage=response.token_usage,
                ),
            )
        else:
            yield event
