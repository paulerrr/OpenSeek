"""
Prompt-based tool calling, modeled on Better DeepSeek's tag protocol.

DeepSeek's web chat has no native function calling, so the tool definitions go into the hidden system prompt
and the model is asked to emit `<tool_call>` blocks, which are parsed back into OpenAI-style tool calls.
"""

import json
import re
import uuid

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
BLOCK_RE = re.compile(re.escape(OPEN_TAG) + r"(.*?)" + re.escape(CLOSE_TAG), re.DOTALL)
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
        call = _parse_call(match.group(1))
        if call is None:
            return match.group(0)
        calls.append(call)
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

    arguments = data.get("arguments", data.get("parameters", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": data["name"], "arguments": arguments},
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
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        output = ""
        while True:
            if self._inside:
                end = self._buffer.find(CLOSE_TAG)
                if end == -1:
                    return output
                self._buffer = self._buffer[end + len(CLOSE_TAG) :]
                self._inside = False
            else:
                start = self._buffer.find(OPEN_TAG)
                if start != -1:
                    output += self._buffer[:start]
                    self._buffer = self._buffer[start + len(OPEN_TAG) :]
                    self._inside = True
                    continue
                hold = _partial_tag_length(self._buffer)
                output += self._buffer[: len(self._buffer) - hold]
                self._buffer = self._buffer[len(self._buffer) - hold :]
                return output

    def finish(self) -> str:
        """Returns held-back text: a partial tag that never completed, or an unterminated block."""
        rest = (OPEN_TAG + self._buffer) if self._inside else self._buffer
        self._buffer, self._inside = "", False
        return rest


def _partial_tag_length(text: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of the opening tag."""
    for length in range(min(len(text), len(OPEN_TAG) - 1), 0, -1):
        if OPEN_TAG.startswith(text[-length:]):
            return length
    return 0
