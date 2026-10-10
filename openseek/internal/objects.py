from dataclasses import dataclass


class Response:
    def __init__(
        self,
        text: str,
        chat_id: str | None = None,
        deepthink_duration: int | None = None,
        deepthink_content: str | None = None,
        search_results: list | None = None,
        message_id: int | None = None,
        token_usage: int | None = None,
    ):
        """
        Response object to store the response from the DeepSeek API.

        Args:
        ----------
            text (str): The response text.
            chat_id (Optional[str], optional): The chat ID. Defaults to None.
            deepthink_duration (Optional[int], optional): The deepthink duration. Defaults to None.
            deepthink_content (Optional[str], optional): The deepthink content. Defaults to None.
            search_results (Optional[list], optional): The search results. Defaults to None.
            message_id (Optional[int], optional): ID of the assistant message within the chat. Defaults to None.
            token_usage (Optional[int], optional): Tokens used by the chat so far, as reported by DeepSeek.
                Defaults to None.
        """
        self.text = text
        self.chat_id = chat_id
        self.deepthink_duration = deepthink_duration
        self.deepthink_content = deepthink_content
        self.search_results = search_results
        self.message_id = message_id
        self.token_usage = token_usage

    def __repr__(self):
        return self.text


@dataclass
class StreamEvent:
    """
    An event yielded by `DeepSeek.stream_message`.

    Attributes:
        type (str): "thinking" or "text" for a content delta, or "done" once the response is complete.
        content (str): The delta text (empty for "done").
        response (Optional[Response]): The complete response, set on the "done" event.
    """

    type: str
    content: str = ""
    response: Response | None = None
