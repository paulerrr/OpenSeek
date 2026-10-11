"""
Prompt-based tool calling, modeled on Better DeepSeek's tag protocol.

DeepSeek's web chat has no native function calling, so the tool definitions go into the hidden system prompt
and the model is asked to emit `<tool_call>` blocks, which are parsed back into OpenAI-style tool calls.
The model sometimes falls back to DeepSeek's native DSML markup instead, so those blocks are accepted too.
"""

import json
import re
import uuid

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
# DeepSeek's native markup: <｜DSML｜function_calls><｜DSML｜invoke name="..."><｜DSML｜parameter name="..." string="true">
# The delimiter is a fullwidth bar, which can come back doubled or as ASCII, so the patterns are lenient
_DSML = r"[｜|]+\s*DSML\s*[｜|]+\s*"
_DSML_OPEN = rf"<{_DSML}(?:function_)?calls\s*>"
_DSML_CLOSE_RE = re.compile(rf"</{_DSML}(?:function_)?calls\s*>")
_INVOKE_RE = re.compile(rf'<{_DSML}invoke\s+name="([^"]*)"\s*>(.*?)</{_DSML}invoke\s*>', re.DOTALL)
_PARAMETER_RE = re.compile(
    rf'<{_DSML}parameter\s+name="([^"]*)"(?:\s+string="(true|false)")?\s*>(.*?)</{_DSML}parameter\s*>', re.DOTALL
)

BLOCK_RE = re.compile(
    rf"{re.escape(OPEN_TAG)}(?P<json>.*?){re.escape(CLOSE_TAG)}|{_DSML_OPEN}(?P<dsml>.*?){_DSML_CLOSE_RE.pattern}",
    re.DOTALL,
)
_OPEN_RE = re.compile(rf"{re.escape(OPEN_TAG)}|{_DSML_OPEN}")
_CLOSE_TAG_RE = re.compile(re.escape(CLOSE_TAG))
# Longest text after a "<" that is held back while streaming in case it starts an opening tag
_MAX_OPEN_TAG_LENGTH = 40
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

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


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Splits a reply into its text content and OpenAI-style tool calls.

    Blocks that don't contain a valid call are left in the text.
    """
    calls = []

    def replace(match: re.Match) -> str:
        if match.group("json") is not None:
            block_calls = [_parse_call(match.group("json"))]
        else:
            block_calls = _parse_dsml_calls(match.group("dsml"))
        if not block_calls or None in block_calls:
            return match.group(0)
        calls.extend(block_calls)
        return ""

    content = re.sub(r"\n{3,}", "\n\n", BLOCK_RE.sub(replace, text))
    return content.strip(), calls


def _parse_call(body: str) -> dict | None:
    body = _FENCE_RE.sub("", body.strip())
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return None

    return _make_call(data["name"], data.get("arguments", data.get("parameters", {})))


def _parse_dsml_calls(body: str) -> list[dict]:
    calls = []
    for name, invoke_body in _INVOKE_RE.findall(body):
        arguments = {}
        for key, is_string, value in _PARAMETER_RE.findall(invoke_body):
            if is_string == "true":
                arguments[key] = value
            else:
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
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

    Text that might be the start of an opening tag is held back until it can be decided.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._open = ""
        self._close_re: re.Pattern | None = None

    def feed(self, text: str) -> str:
        self._buffer += text
        output = ""
        while True:
            if self._close_re:
                end = self._close_re.search(self._buffer)
                if end is None:
                    return output
                self._buffer = self._buffer[end.end() :]
                self._open, self._close_re = "", None
            else:
                start = _OPEN_RE.search(self._buffer)
                if start:
                    output += self._buffer[: start.start()]
                    self._buffer = self._buffer[start.end() :]
                    self._open = start.group(0)
                    self._close_re = _CLOSE_TAG_RE if self._open == OPEN_TAG else _DSML_CLOSE_RE
                    continue
                hold = _partial_tag_length(self._buffer)
                output += self._buffer[: len(self._buffer) - hold]
                self._buffer = self._buffer[len(self._buffer) - hold :]
                return output

    def finish(self) -> str:
        """Returns held-back text: a partial tag that never completed, or an unterminated block."""
        rest = self._open + self._buffer
        self._buffer, self._open, self._close_re = "", "", None
        return rest


def _partial_tag_length(text: str) -> int:
    """Length of the trailing text from the last "<" that has no ">" yet, since it might start an opening tag."""
    start = text.rfind("<")
    if start == -1 or ">" in text[start:] or len(text) - start > _MAX_OPEN_TAG_LENGTH:
        return 0
    return len(text) - start
