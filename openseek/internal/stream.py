"""Parser for DeepSeek's chat completion SSE stream."""

import json

THINK = "THINK"
RESPONSE = "RESPONSE"


class SSEParser:
    """Incrementally splits raw SSE text into (event, data) pairs."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str | None, str]]:
        self._buffer += text.replace("\r\n", "\n")
        events = []
        while "\n\n" in self._buffer:
            raw, self._buffer = self._buffer.split("\n\n", 1)
            event, data = None, []
            for line in raw.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            if data:
                events.append((event, "\n".join(data)))
        return events

    @property
    def leftover(self) -> str:
        return self._buffer


class ResponseState:
    """
    Rebuilds the assistant message from DeepSeek's patch operations.

    Each data line is `{"p": path, "o": op, "v": value}`. `p` and `o` are omitted when they repeat the
    previous line's, so `{"v": "text"}` appends to the same fragment as before. Ops are SET (default),
    APPEND and BATCH (a list of relative `{"p", "v"}` operations).
    """

    def __init__(self) -> None:
        self.root: dict = {"response": {"fragments": []}}
        self.message_id: int | None = None
        # Message of an error hint, such as DeepSeek's rate limit, which replaces the response
        self.error: str | None = None
        self._path: str | None = None
        self._op: str | None = None

    @property
    def response(self) -> dict:
        return self.root.get("response") or {}

    @property
    def fragments(self) -> list[dict]:
        return self.response.get("fragments") or []

    def feed(self, event: str | None, data: str) -> list[tuple[str, str]]:
        """Applies one SSE event, returning the (fragment type, text) deltas it produced."""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return []

        if event == "ready":
            self.message_id = payload.get("response_message_id")
            return []
        if event == "hint" and isinstance(payload, dict) and payload.get("type") == "error":
            self.error = str(payload.get("content") or "DeepSeek returned an error")
            return []
        if event is not None or not isinstance(payload, dict) or "v" not in payload:
            return []

        if "p" in payload:
            self._path = payload["p"]
            self._op = payload.get("o", "SET")
        elif "o" in payload:
            self._op = payload["o"]

        if self._path is None:
            # Initial snapshot of the whole message
            if isinstance(payload["v"], dict):
                self.root.update(payload["v"])
                return [(f.get("type"), f.get("content") or "") for f in self.fragments if f.get("content")]
            return []

        return self._apply(self._path, self._op or "SET", payload["v"])

    def _apply(self, path: str, op: str, value) -> list[tuple[str, str]]:
        if op == "BATCH":
            deltas = []
            for item in value or []:
                if isinstance(item, dict) and "p" in item:
                    deltas += self._apply(f"{path}/{item['p']}", item.get("o", "SET"), item.get("v"))
            return deltas

        parts = path.split("/")
        parent = self.root
        for part in parts[:-1]:
            child = _get(parent, part)
            if child is None:
                child = {}
                _set(parent, part, child)
            parent = child
        key = parts[-1]

        if op == "APPEND":
            current = _get(parent, key)
            if isinstance(current, list):
                current.extend(value if isinstance(value, list) else [value])
            else:
                _set(parent, key, (current or "") + str(value))
        else:
            _set(parent, key, value)

        # Text deltas: appended fragment content, or newly appended fragments
        if key == "content" and len(parts) >= 3 and parts[-3] == "fragments" and isinstance(parent, dict):
            return [(parent.get("type"), str(value))] if op == "APPEND" else []
        if key == "fragments" and op == "APPEND":
            new = value if isinstance(value, list) else [value]
            return [(f.get("type"), f.get("content") or "") for f in new if isinstance(f, dict) and f.get("content")]
        return []

    def text_of(self, fragment_type: str) -> str:
        return "".join(f.get("content") or "" for f in self.fragments if f.get("type") == fragment_type)


def _get(container, key: str):
    if isinstance(container, list):
        try:
            return container[int(key)]
        except (ValueError, IndexError):
            return None
    return container.get(key) if isinstance(container, dict) else None


def _set(container, key: str, value) -> None:
    if isinstance(container, list):
        container[int(key)] = value
    elif isinstance(container, dict):
        container[key] = value
