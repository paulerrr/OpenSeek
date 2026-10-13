"""
Prompt-based tool calling, modeled on Better DeepSeek's tag protocol.

DeepSeek's web chat has no native function calling, so the tool definitions go into the hidden system prompt
and the model is asked to emit `<tool_call>` blocks, which are parsed back into OpenAI-style tool calls.
The model sometimes falls back to DeepSeek's native DSML markup instead, so those blocks are accepted too, and so are
bare `{"name": ..., "arguments": ...}` objects written on their own line without tags, when the name is one of the
request's tools and the object isn't inside a code fence.
"""

import json
import re
import uuid

from .exceptions import ServerDown

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
# DeepSeek's native markup: <｜DSML｜function_calls><｜DSML｜invoke name="..."><｜DSML｜parameter name="..." string="true">
# The delimiter is a fullwidth bar, which can come back doubled or as ASCII, so the patterns are lenient
_DSML = r"[｜|]+\s*DSML\s*[｜|]+\s*"
_DSML_OPEN = rf"<{_DSML}(?:function_)?calls\s*>"
_DSML_CLOSE_RE = re.compile(rf"</{_DSML}(?:function_)?calls\s*>")
# Calls may lack an outer wrapper, or have an extra slash before an invoke with a name attribute.
# Inner closing tags sometimes come back without the DSML delimiter (</invoke>).
_INVOKE_OPEN = rf'</?{_DSML}invoke\s+name="[^"]*"\s*>'
_INVOKE_CLOSE = rf'</(?:{_DSML})?invoke\s*>'
_INVOKE_RE = re.compile(rf'</?{_DSML}invoke\s+name="([^"]*)"\s*>(.*?){_INVOKE_CLOSE}', re.DOTALL)
_PARAMETER_RE = re.compile(
    rf'<{_DSML}parameter\s+name="([^"]*)"(?:\s+string="(true|false)")?\s*>(.*?)</(?:{_DSML})?parameter\s*>',
    re.DOTALL,
)

# The model sometimes misspells the closing tag (</call_call>, </call-tool>) or omits it before the next block,
# or closes it with DSML markup (</｜DSML｜invoke>), so a block ends at any closing tag containing "call", a DSML
# invoke or calls closing tag, just before the next opening tag (including a bare DSML invoke), or at the end
_OPEN = re.escape(OPEN_TAG)
_CLOSE = (
    rf"</\s*[A-Za-z_-]*call[A-Za-z_-]*\s*>|</{_DSML}(?:(?:function_)?calls|invoke)\s*>"
    rf"|(?={_OPEN}|{_DSML_OPEN}|{_INVOKE_OPEN})"
)
BLOCK_RE = re.compile(
    rf"{_OPEN}(?P<json>(?:(?!{_OPEN}).)*?)(?:{_CLOSE}|\Z)"
    rf"|{_DSML_OPEN}(?P<dsml>.*?){_DSML_CLOSE_RE.pattern}"
    rf"|(?P<invoke>{_INVOKE_OPEN}.*?{_INVOKE_CLOSE})",
    re.DOTALL,
)
# An empty DSML opener at end of output is discarded only after a validated call.
_TRAILING_DSML_OPEN_RE = re.compile(rf"\s*{_DSML_OPEN}\s*\Z")
_OPEN_RE = re.compile(rf"{_OPEN}|{_DSML_OPEN}|{_INVOKE_OPEN}")
_CLOSE_TAG_RE = re.compile(_CLOSE)
# Stray closing markup is discarded only immediately after a successfully parsed call.
_ORPHAN_CLOSE_RE = re.compile(rf'</{_DSML}(?:(?:function_)?calls|parameter|invoke)\s*>')
# Longest text after a "<" that is held back while streaming in case it starts an opening tag
_MAX_OPEN_TAG_LENGTH = 256
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
# A bare call is a JSON object that starts with its "name" key
_BARE_START_RE = re.compile(r'\{\s*"name"\s*:')
_INCOMPLETE = object()

_NAMELESS_PARAMETER_RE = re.compile(rf'<{_DSML}parameter\s*>')


class ToolOutputGuard:
    """Hold nameless parameter runs until they resolve; reject four consecutive tags.

    This is deliberately narrow: ordinary malformed text and fenced examples remain
    text. Chunk boundaries must not change whether a runaway response is detected.
    """

    def __init__(self):
        self.buffer = ""
        self.pending = ""
        self.count = 0
        self.line = ""
        self.in_fence = False

    def _emit(self, text):
        self.line += text
        while "\n" in self.line:
            line, self.line = self.line.split("\n", 1)
            if line.lstrip().startswith(("```", "~~~")):
                self.in_fence = not self.in_fence
        return text

    def feed(self, text):
        self.buffer += text
        output = []
        while self.buffer:
            match = _NAMELESS_PARAMETER_RE.search(self.buffer)
            newline = self.buffer.find("\n")
            if match and not self.in_fence and (newline < 0 or match.start() < newline):
                before = self.buffer[:match.start()]
                if before.strip():
                    output.append(self._emit(self.pending + before))
                    self.pending, self.count = "", 0
                else:
                    self.pending += before
                self.pending += match.group()
                self.count += 1
                if self.count >= 4:
                    raise ServerDown("DeepSeek generated repeated malformed DSML parameter tags; retry the request.")
                self.buffer = self.buffer[match.end():]
                continue
            end = newline + 1 if newline >= 0 else len(self.buffer) - _partial_tag_length(self.buffer)
            if not end:
                break
            part, self.buffer = self.buffer[:end], self.buffer[end:]
            if self.pending and not part.strip():
                self.pending += part
            else:
                output.append(self._emit(self.pending + part))
                self.pending, self.count = "", 0
        return "".join(output)

    def finish(self):
        rest = self.pending + self.buffer
        self.pending, self.buffer, self.count = "", "", 0
        return rest

# After its calls the model sometimes keeps going and writes the next input turn itself: a closing
# </system_instructions> and invented <tool_result> (or <result>) blocks, which it then reasons from
_FAKE_RESULT_STARTS = ("</system_instructions>", "<tool_result name=", "<tool_result>", "<result name=", "<result>")


class FabricatedResultGuard:
    """Cut a reply where a fabricated tool result starts after a valid tool call.

    Only a marker at the start of a line outside code fences counts, and only once the text before it holds a
    call for one of `tool_names`, so replies that merely mention the format are left alone. Text that might
    still become a marker is held back. After `cut` is set, everything else is dropped.
    """

    def __init__(self, tool_names=None):
        self.tool_names = tool_names
        self.text = ""
        self.buffer = ""
        self.line = ""
        self.mid_line = False
        self.in_fence = False
        self.cut = False

    def feed(self, text: str) -> str:
        if self.cut:
            return ""
        self.buffer += text
        output = ""
        while self.buffer:
            newline = self.buffer.find("\n")
            line = self.buffer if newline == -1 else self.buffer[: newline + 1]
            if not self.mid_line and not self.in_fence:
                start = line.lstrip(" \t")
                if start.startswith(_FAKE_RESULT_STARTS) and parse_tool_calls(self.text, self.tool_names)[1]:
                    self.cut, self.buffer = True, ""
                    return output
                if newline == -1 and any(marker.startswith(start) for marker in _FAKE_RESULT_STARTS):
                    break
            self.buffer = self.buffer[len(line) :]
            self.text += line
            self.line += line
            output += line
            self.mid_line = newline == -1
            if not self.mid_line:
                if self.line.lstrip().startswith(("```", "~~~")):
                    self.in_fence = not self.in_fence
                self.line = ""
        return output

    def finish(self) -> str:
        rest, self.buffer = self.buffer, ""
        self.text += rest
        return rest


TOOLS_REMINDER = (
    f"Reminder: call tools only with {OPEN_TAG}{{\"name\": ..., \"arguments\": {{...}}}}{CLOSE_TAG} blocks, then stop "
    "and wait for the results."
)


def build_tools_prompt(tools: list[dict], tool_choice=None) -> str:
    """Describes the available tools and the calling protocol."""
    definitions = []
    for tool in tools:
        function = tool.get("function", tool)
        definitions.append(
            {
                "name": function.get("name"),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )

    if tool_choice == "required":
        choice = "You must call at least one tool in your reply."
    elif isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
        choice = f"You must call the `{tool_choice['function']['name']}` tool in your reply."
    else:
        choice = "Call tools only when they help; otherwise answer normally."

    return f"""# Tools

You can call the tools defined below. To call one, write a block exactly like this, with a single JSON object
whose "arguments" match the tool's parameters schema:

{OPEN_TAG}
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
{CLOSE_TAG}

Rules:
- Write tool call blocks as plain text, never inside code fences.
- You may call several tools in one reply by writing several blocks.
- After your tool call blocks, stop and wait. The results come back in the next message as
  <tool_result> blocks. Never invent tool results.
- Only call tools listed below. {choice}

Available tools (JSON):
{json.dumps(definitions, indent=2)}"""


def parse_tool_calls(text: str, tool_names=None) -> tuple[str, list[dict]]:
    """
    Splits a reply into its text content and OpenAI-style tool calls.

    Blocks that don't contain a valid call are left in the text. Bare JSON calls are accepted only for `tool_names`.
    """
    spans = []
    for match in BLOCK_RE.finditer(text):
        if match.group("json") is not None:
            block_calls = [_parse_call(match.group("json"))]
        else:
            body = match.group("dsml") if match.group("dsml") is not None else match.group("invoke")
            block_calls = _parse_dsml_calls(body)
        valid = bool(block_calls) and all(
            call is not None and (tool_names is None or call["function"]["name"] in tool_names)
            for call in block_calls
        )
        spans.append((match.start(), match.end(), block_calls if valid else None))
    if tool_names:
        spans += _bare_call_spans(text, [(start, end) for start, end, _ in spans], tool_names)
    spans.sort(key=lambda span: span[0])

    calls, parts, position = [], [], 0
    for start, end, span_calls in spans:
        if span_calls is None:
            continue
        parts.append(text[position:start])
        calls.extend(span_calls)
        position = end
        while True:
            remaining = text[position:]
            closing = _ORPHAN_CLOSE_RE.match(text, position + len(remaining) - len(remaining.lstrip()))
            if closing is None:
                break
            position = closing.end()
    if calls and _TRAILING_DSML_OPEN_RE.fullmatch(text[position:]):
        position = len(text)
    parts.append(text[position:])
    content = re.sub(r"\n{3,}", "\n\n", "".join(parts))
    return content.strip(), calls


def _bare_call_spans(text: str, blocks: list[tuple[int, int]], tool_names) -> list[tuple[int, int, list[dict]]]:
    """Finds bare JSON calls at the start of lines outside tag blocks and code fences."""
    spans, in_fence, line_start = [], False, 0
    while line_start <= len(text):
        block = next(((start, end) for start, end in blocks if start <= line_start < end), None)
        if block:
            # Text after a block on its last line is skipped, as the stream filter only checks line starts
            line_start = text.find("\n", block[1])
            line_start = len(text) + 1 if line_start == -1 else line_start + 1
            continue
        line_end = text.find("\n", line_start)
        line_end = len(text) if line_end == -1 else line_end
        line = text[line_start:line_end]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.lstrip(" \t").startswith("{"):
            brace = line_start + len(line) - len(line.lstrip(" \t"))
            found = _match_bare_call(text, brace, tool_names, final=True)
            if found and not any(brace < end and start < found[0] for start, end in blocks):
                spans.append((brace, found[0], [found[1]]))
                line_end = text.find("\n", found[0])
                line_end = len(text) if line_end == -1 else line_end
        line_start = line_end + 1
    return spans


def _match_bare_call(text: str, start: int, tool_names, final: bool):
    """
    Checks for a bare call whose object starts at `start` and ends its line.

    Returns (end, call), None when the text isn't a call, or `_INCOMPLETE` when more text is needed to decide.
    """
    if not _BARE_START_RE.match(text, start):
        return None
    end = _object_end(text, start)
    if end is None:
        return None if final else _INCOMPLETE
    newline = text.find("\n", end)
    if newline == -1 and not final:
        return _INCOMPLETE
    if text[end : len(text) if newline == -1 else newline].strip():
        return None
    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return None
    if data.get("name") not in tool_names or not isinstance(data.get("arguments"), (dict, str)):
        return None
    return end, _make_call(data["name"], data["arguments"])


def _object_end(text: str, start: int) -> int | None:
    """Index just past the brace that closes the object starting at `start`, or None if it doesn't close."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _parse_call(body: str) -> dict | None:
    body = _FENCE_RE.sub("", body.strip())
    try:
        data, end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        return None
    if _ORPHAN_CLOSE_RE.sub("", body[end:]).strip():
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None

    arguments = data.get("arguments", data.get("parameters", {}))
    if not isinstance(arguments, (dict, str)):
        return None
    return _make_call(data["name"], arguments)


def _parse_dsml_calls(body: str) -> list[dict]:
    calls = []
    for name, invoke_body in _INVOKE_RE.findall(body):
        # Never turn unparsed or nested parameter markup into an empty/partial call.
        if _PARAMETER_RE.sub("", invoke_body).strip():
            return []
        arguments = {}
        for key, is_string, value in _PARAMETER_RE.findall(invoke_body):
            if key in arguments or re.search(rf'</?{_DSML}', value):
                return []
            if is_string == "true":
                arguments[key] = value
            else:
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
        # The model sometimes passes the whole arguments object as one parameter named "arguments"
        if list(arguments) == ["arguments"] and isinstance(arguments["arguments"], dict):
            arguments = arguments["arguments"]
        calls.append(_make_call(name, arguments))
    return calls


def _make_call(name: str, arguments) -> dict:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def format_tool_call(call: dict) -> str:
    """Renders an OpenAI tool call back into the tag format, for replaying history."""
    function = call.get("function", {})
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = function.get("arguments")
    return f"{OPEN_TAG}\n{json.dumps({'name': function.get('name'), 'arguments': arguments})}\n{CLOSE_TAG}"


def format_tool_result(content: str, tool_call_id: str | None, name: str | None) -> str:
    attributes = "".join(f' {key}="{value}"' for key, value in (("name", name), ("id", tool_call_id)) if value)
    return f"<tool_result{attributes}>\n{content}\n</tool_result>"


class ToolCallStreamFilter:
    """
    Removes tool call blocks from streamed text, so only plain content is forwarded while streaming.

    Text that might be the start of an opening tag, or of a bare call for one of `tool_names`, is held back until it
    can be decided.
    """

    def __init__(self, tool_names=None) -> None:
        self._tool_names = tool_names
        self._buffer = ""
        self._open = ""
        self._close_re: re.Pattern | None = None
        # The current line outside tag blocks, and whether it is inside a code fence, for spotting bare calls
        self._line = ""
        self._in_fence = False
        # After a tag block, the rest of its line is never checked for a bare call, matching parse_tool_calls
        self._after_block = False
        self._after_call = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output = ""
        while True:
            if self._close_re:
                end = self._close_re.search(self._buffer)
                if end is None:
                    return output
                block = self._open + self._buffer[: end.end()]
                self._after_call = bool(parse_tool_calls(block, self._tool_names)[1])
                if not self._after_call:
                    output += block
                self._buffer = self._buffer[end.end() :]
                self._open, self._close_re = "", None
                self._after_block = True
                continue
            if self._after_call:
                closing = _ORPHAN_CLOSE_RE.match(self._buffer.lstrip())
                if closing:
                    offset = len(self._buffer) - len(self._buffer.lstrip())
                    self._buffer = self._buffer[offset + closing.end():]
                    continue
            bare = self._bare_call()
            if bare is _INCOMPLETE:
                return output
            if bare:
                # Drops the call along with the line's indentation
                self._buffer = self._buffer[bare[0] :]
                self._after_call = True
                self._after_block = True
                continue
            start = _OPEN_RE.search(self._buffer)
            newline = self._buffer.find("\n")
            if start and (newline == -1 or start.start() < newline):
                output += self._emit(self._buffer[: start.start()])
                self._buffer = self._buffer[start.end() :]
                self._open = start.group(0)
                if self._open == OPEN_TAG:
                    self._close_re = _CLOSE_TAG_RE
                elif re.fullmatch(_INVOKE_OPEN, self._open):
                    self._close_re = re.compile(_INVOKE_CLOSE)
                else:
                    self._close_re = _DSML_CLOSE_RE
                continue
            if newline != -1:
                # Lines are passed one at a time, so the start of the next one can be checked for a bare call
                output += self._emit(self._buffer[: newline + 1])
                self._buffer = self._buffer[newline + 1 :]
                continue
            hold = max(_partial_tag_length(self._buffer), self._partial_bare_length())
            output += self._emit(self._buffer[: len(self._buffer) - hold])
            self._buffer = self._buffer[len(self._buffer) - hold :]
            return output

    def finish(self) -> str:
        """Returns held-back text: a partial tag that never completed, or an unterminated block that isn't a call."""
        rest = self._open + self._buffer
        bare = None if self._open else self._bare_call(final=True)
        self._buffer, self._open, self._close_re = "", "", None
        if bare or (self._after_call and _TRAILING_DSML_OPEN_RE.fullmatch(rest)):
            return ""
        content, calls = parse_tool_calls(rest, self._tool_names)
        return content if calls else rest

    def _emit(self, text: str) -> str:
        """Tracks the current line and code fences in text passed through outside tag blocks."""
        if text.strip():
            self._after_call = False
        self._line += text
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            self._after_block = False
            if line.lstrip().startswith("```"):
                self._in_fence = not self._in_fence
        return text

    def _at_line_start(self) -> bool:
        return bool(self._tool_names) and not self._in_fence and not self._after_block and not self._line.strip(" \t")

    def _bare_call(self, final: bool = False):
        """Checks whether the buffer, at the start of a line outside code fences, begins with a bare call."""
        if not self._at_line_start():
            return None
        indent = len(self._buffer) - len(self._buffer.lstrip(" \t"))
        if indent == len(self._buffer) or self._buffer[indent] != "{":
            return None
        return _match_bare_call(self._buffer, indent, self._tool_names, final)

    def _partial_bare_length(self) -> int:
        """Length of trailing text that might still become the start of a bare call."""
        if not self._at_line_start():
            return 0
        candidate = self._buffer.lstrip(" \t")
        if not candidate:
            return len(self._buffer)
        rest = candidate[1:].lstrip()
        if candidate[0] == "{" and ('"name"'.startswith(rest) or (rest.startswith('"name"') and not rest[6:].strip())):
            return len(self._buffer)
        return 0


def _partial_tag_length(text: str) -> int:
    """Length of the trailing text from the last "<" that has no ">" yet, since it might start an opening tag."""
    start = text.rfind("<")
    if start == -1 or ">" in text[start:] or len(text) - start > _MAX_OPEN_TAG_LENGTH:
        return 0
    return len(text) - start
