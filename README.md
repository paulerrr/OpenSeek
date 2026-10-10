**OpenSeek** is an open-source Python library that provides free access to DeepSeek models. Designed for developers and
researchers, it offers a simple and unified interface to interact with various DeepSeek LLM models without requiring API
keys, subscriptions, or paid plans.

## Installation

```cmd
pip install openseek
```

## Quick Start

### Asynchronous Usage

```python
import asyncio
from openseek import DeepSeek


async def main():
    async with DeepSeek(
            email="your_email@example.com",
            password="your_password"
    ) as api:
        response = await api.send_message("Tell me about quantum physics")
        print(response.text)


asyncio.run(main())
```

### Synchronous Usage

```python
from openseek import DeepSeek


def main():
    api = DeepSeek(
        email="your_email@example.com",
        password="your_password"
    )

    try:
        api.initialize_sync()
        response = api.send_message_sync("Tell me about quantum physics")
        print(response.text)
    finally:
        api.close_sync()


main()
```

## Features

- **Free Access**: Use DeepSeek models without API keys or paid subscriptions
- **Asynchronous and Synchronous API**: Flexibility for different usage scenarios
- **DeepThink Mode**: Activate advanced thinking features for more complex queries
- **Search Support**: Enable web search to get up-to-date information
- **Session Management**: Resume existing chats using a session ID
- **Streaming**: Receive DeepThink reasoning and the answer as they are generated
- **Hidden System Prompts**: Instructions are added to the request payload, not typed into the chat
- **Attachments**: Send images, PDFs, Office documents and code files with a message
- **OpenAI-Compatible Server**: `server.py` serves `/v1/chat/completions` with streaming and tool calling

## Parameters

When initializing `DeepSeek`, the following parameters are available:

| Parameter         | Type             | Default | Description                             |
|-------------------|------------------|---------|-----------------------------------------|
| email             | Optional\[str\]  | None    | Email for logging into DeepSeek         |
| password          | Optional\[str\]  | None    | Password for logging into DeepSeek      |
| token             | Optional\[str\]  | None    | User token; skips login and email verification |
| user_data_dir     | Optional\[str\]  | None    | Persistent Chrome profile directory     |
| headless          | bool             | True    | Launch browser in headless mode         |
| verbose           | bool             | False   | Enable detailed logging                 |
| chrome_args       | Optional\[List\] | None    | Additional arguments for Chrome         |
| attempt_cf_bypass | bool             | True    | Attempt to bypass Cloudflare protection |

## Sending Messages

The `send_message` method accepts:

| Parameter       | Type  | Default | Description                                     |
|-----------------|-------|---------|-------------------------------------------------|
| message         | str   | -       | Message text to send                            |
| slow_mode       | bool  | False   | Send the message slowly, character by character |
| deepthink       | bool  | False   | Activate DeepThink feature                      |
| search          | bool  | False   | Activate search feature                         |
| timeout         | int   | 60      | Maximum response waiting time                   |
| slow_mode_delay | float | 0.25    | Delay between characters in slow\_mode          |
| chat_id         | str   | None    | ID of an existing chat to continue              |
| system_prompt   | str   | None    | Hidden instructions sent with the message       |
| files           | list  | None    | Paths of files to attach (images, documents...) |

The returned `Response` has `text`, `chat_id`, `deepthink_content`, `deepthink_duration`, `search_results`,
`message_id` and `token_usage`.

### Streaming

`stream_message` takes the same arguments and yields events as the response is generated:

```python
async for event in api.stream_message("Explain tides", deepthink=True):
    if event.type == "thinking":
        print(event.content, end="")   # DeepThink reasoning
    elif event.type == "text":
        print(event.content, end="")   # the answer
    elif event.type == "done":
        response = event.response      # the complete Response
```

Replies are read from DeepSeek's own response stream rather than the rendered page, so the text is the model's
raw markdown.

### Getting a token

Password login can be blocked by DeepSeek's email verification step. To avoid it, log in at
chat.deepseek.com in a browser, open the developer console and run
`JSON.parse(localStorage.userToken).value`, then pass the result as `token`.

## OpenAI-Compatible Server

`server.py` exposes the library as an OpenAI-style API on port 8888 (or run it with `docker compose up`):

```bash
pip install fastapi uvicorn
DEEPSEEK_TOKEN=<token> python server.py
```

- `POST /v1/chat/completions` with `stream` support. Use model `deepseek-reasoner` (or `"deepthink": true`) for
  DeepThink, whose reasoning is returned as `reasoning_content`; `"search": true` enables web search.
- `system` / `developer` messages become a hidden system prompt.
- Attachments: `image_url` parts (data URLs or http(s) URLs) and `file` parts (`file_data` + `filename`) are
  uploaded with the message, up to 50 files of 100 MB each. A replayed conversation re-uploads its earlier ones.
- `tools` and `tool_choice` work through a prompt protocol: the tools are described to the model, and the
  `<tool_call>` blocks it writes are returned as `tool_calls`. Send results back as `tool` messages as usual.
- Each conversation is matched to its DeepSeek chat by its message history, so several clients can be
  served at once. A conversation the server hasn't seen (for example after a restart) is replayed into a
  new chat.

Environment variables: `DEEPSEEK_TOKEN` (or `DEEPSEEK_EMAIL` and `DEEPSEEK_PASSWORD`), `OPENSEEK_PORT`,
`OPENSEEK_TIMEOUT` (seconds, default 300) and `OPENSEEK_USER_DATA_DIR`.

## Error Handling

The library can raise the following exceptions:

- `MissingCredentials`: Missing credentials (token, or email and password)
- `VerificationRequired`: DeepSeek asked for email verification during password login
- `FileUploadError`: An attached file is missing, of an unsupported type, or failed to process
- `InvalidCredentials`: Invalid credentials
- `ServerDown`: DeepSeek server is unavailable
- `MissingInitialization`: Required dependencies are missing or initialization not done

## Requirements

- Python 3.10 or higher
- Dependencies: beautifulsoup4, zendriver, inscriptis, pyvirtualdisplay (for Linux)

## License

MIT - see [LICENSE](LICENSE) for details.

## Author

Daniel Cuzneţov <danielcuznetov04@gmail.com>
