import asyncio
import contextlib
import json
import logging
import os
import platform
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import zendriver
from zendriver import cdp

from .internal.exceptions import (
    FileUploadError,
    OpenSeekException,
    InvalidCredentials,
    MissingCredentials,
    MissingInitialization,
    RateLimited,
    ServerDown,
    VerificationRequired,
)
from .internal.interceptor import BINDING_NAME, INTERCEPTOR_JS
from .internal.objects import Response, StreamEvent
from .internal.selectors import DeepSeekSelectors
from .internal.stream import RESPONSE, THINK, ResponseState, SSEParser

BASE_URL = "https://chat.deepseek.com/"
# File states DeepSeek reports once processing has ended badly
FAILED_FILE_STATUSES = ("FAILED", "ERROR", "CANCELED", "CANCELLED")


class DeepSeek:
    def __init__(
        self,
        email: str | None = None,
        password: str | None = None,
        token: str | None = None,
        headless: bool = True,
        verbose: bool = False,
        chrome_args: list[str] | None = None,
        attempt_cf_bypass: bool = True,
        user_data_dir: str | None = None,
        browser_timeout: float = 20.0,
    ) -> None:
        """
        DeepSeek API automation class.

        Args:
            email (Optional[str]): User email for DeepSeek login.
            password (Optional[str]): User password for DeepSeek login.
            token (Optional[str]): DeepSeek user token (localStorage `userToken`). Skips the login form
                and email verification when provided.
            headless (bool): Whether to run browser in headless mode.
            verbose (bool): Enable detailed logging output.
            chrome_args (Optional[List[str]]): Additional arguments for Chrome.
            attempt_cf_bypass (bool): Whether to bypass Cloudflare protection.
            user_data_dir (Optional[str]): Persistent Chrome profile directory, so a logged-in session
                survives restarts.
            browser_timeout (float): Seconds each browser command may take before the browser is replaced.
                Typing a long message gets extra time on top of this.

        Raises:
            MissingCredentials: If neither a token nor email and password are provided.
        """
        if not token and not (email and password):
            raise MissingCredentials("A token, or email and password, are required for login.")

        self.email = email
        self.password = password
        self.token = token
        self.headless = headless
        self.verbose = verbose
        self.chrome_args = chrome_args or []
        self.attempt_cf_bypass = attempt_cf_bypass
        self.user_data_dir = user_data_dir

        self._initialized = False
        self._healthy = False
        self._recovery_task = None
        self._keepalive_task = None
        self.browser_timeout = browser_timeout
        self.recovery_timeout = 60.0
        # Serializes browser interactions so the keep-alive reload never interrupts a response
        self._lock = asyncio.Lock()
        # Interceptor events for the in-flight request, matched by token
        self._events: asyncio.Queue = asyncio.Queue()
        self._active_token: str | None = None
        # Chat of the latest sent message, known as soon as the request goes out, even if its stream is cut short
        self.active_chat_id: str | None = None
        self.selectors = DeepSeekSelectors()

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Sets up the internal logger."""
        logger = logging.getLogger("DeepSeek")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        return logger

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self) -> None:
        """
        Initializes browser session and logs in to DeepSeek.

        Raises:
            MissingInitialization: When dependencies or environment is not set up correctly.
        """
        if platform.system() == "Linux" and "DISPLAY" not in os.environ:
            self.logger.debug("Starting virtual display...")
            try:
                from pyvirtualdisplay import Display

                self.display = Display()
                self.display.start()
            except ModuleNotFoundError as e:
                raise MissingInitialization("Install PyVirtualDisplay: `pip install pyvirtualdisplay`") from e
            except FileNotFoundError as e:
                raise MissingInitialization("Install Xvfb: `sudo apt install xvfb`") from e

        await asyncio.wait_for(self._launch_browser(), self.recovery_timeout)
        self._initialized = True
        self._keepalive_task = asyncio.create_task(self._keep_alive())

    async def _launch_browser(self):
        self.browser = await zendriver.start(
            chrome_args=self.chrome_args, headless=self.headless, user_data_dir=self.user_data_dir
        )
        await self._install_interceptor()
        await self.browser.get(BASE_URL)

        self.logger.debug("Waiting for page to load...")
        await self._wait_for_any(
            [self.selectors.interactions.textbox, self.selectors.login.email_input], timeout=20
        )

        await self._login()
        self._healthy = True

    async def _browser_call(self, operation, extra: float = 0):
        """Bound CDP commands even when the renderer stops answering."""
        return await asyncio.wait_for(operation, self.browser_timeout + extra)

    async def _stop_browser(self):
        browser = getattr(self, "browser", None)
        if browser is None:
            return
        try:
            await asyncio.wait_for(browser.stop(), 5)
        except (Exception, asyncio.CancelledError) as exc:
            # Browser.stop itself can hang on CDP. Kill the owned process as a fallback.
            process = getattr(browser, "_process", None)
            if process is not None and process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await asyncio.wait_for(asyncio.to_thread(process.wait), 5)
            if isinstance(exc, asyncio.CancelledError):
                raise

    def _schedule_recovery(self):
        self._healthy = False
        self._active_token = None
        self.active_chat_id = None
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recover_browser())

    async def _recover_browser(self):
        self.logger.warning("Replacing unresponsive or interrupted DeepSeek browser")
        try:
            await self._stop_browser()
            await asyncio.wait_for(self._launch_browser(), self.recovery_timeout)
            self.logger.info("DeepSeek browser recovered")
        except (Exception, OpenSeekException):
            self._healthy = False
            self.logger.exception("DeepSeek browser recovery failed")
            await self._stop_browser()

    async def _ensure_browser(self):
        # Call while holding _lock. A timed-out client must not cancel shared recovery.
        if not self._healthy:
            self._schedule_recovery()
            await asyncio.shield(self._recovery_task)
            if not self._healthy:
                raise ServerDown("DeepSeek browser recovery failed; retry later")

    async def _keep_alive(self):
        while self._initialized:
            await asyncio.sleep(300)
            async with self._lock:
                if not self._initialized:
                    break
                try:
                    await self._ensure_browser()
                    await self._browser_call(self.browser.main_tab.reload())
                    await self._browser_call(
                        self.browser.main_tab.wait_for(self.selectors.interactions.textbox, timeout=15)
                    )
                except (Exception, OpenSeekException):
                    self.logger.warning("Browser keep-alive failed", exc_info=True)
                    self._schedule_recovery()

    async def _install_interceptor(self) -> None:
        """Installs the request interceptor on every page load and routes its events to `_events`."""
        tab = self.browser.main_tab
        tab.add_handler(cdp.runtime.BindingCalled, self._on_binding_called)
        await tab.send(cdp.runtime.enable())
        await tab.send(cdp.page.enable())
        await tab.send(cdp.runtime.add_binding(BINDING_NAME))
        await tab.send(cdp.page.add_script_to_evaluate_on_new_document(INTERCEPTOR_JS))

    def _on_binding_called(self, event: cdp.runtime.BindingCalled) -> None:
        if event.name != BINDING_NAME:
            return
        try:
            message = json.loads(event.payload)
        except json.JSONDecodeError:
            return
        if self._active_token and message.get("token") == self._active_token:
            self._events.put_nowait(message)

    async def _js(self, expression: str):
        """Evaluates a JavaScript expression in the main tab."""
        return await self._browser_call(self.browser.main_tab.evaluate(expression))

    async def _exists(self, selector: str) -> bool:
        return bool(await self._js(f"!!document.querySelector({json.dumps(selector)})"))

    async def _wait_for_any(self, selectors: list[str], timeout: float) -> str | None:
        """Waits until one of the selectors matches, returning it, or None on timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            for selector in selectors:
                with contextlib.suppress(Exception):
                    if await self._exists(selector):
                        return selector
            await asyncio.sleep(0.5)
        return None

    async def _login(self):
        """Logs in to DeepSeek using the token, an existing profile session, or email and password."""
        if await self._exists(self.selectors.interactions.textbox):
            self.logger.info("Already logged into DeepSeek.")
            return

        if self.token:
            await self._login_with_token()
        else:
            await self._login_with_password()

    async def _login_with_token(self):
        """Injects the user token into localStorage and reloads."""
        self.logger.debug("Logging in with token...")
        value = json.dumps(json.dumps({"value": self.token, "__version": "0"}))
        await self._js(f"localStorage.setItem('userToken', {value})")
        await self.browser.main_tab.get(BASE_URL)

        if not await self._wait_for_any([self.selectors.interactions.textbox], timeout=20):
            raise InvalidCredentials("Login with token failed, the token may be invalid or expired.")
        self.logger.info("Successfully logged into DeepSeek.")

    async def _login_with_password(self):
        """Fills in the login form with email and password."""
        self.logger.debug("Logging in with email and password...")
        tab = self.browser.main_tab

        with contextlib.suppress(Exception):
            await (await tab.select(self.selectors.login.cookie_dismiss_button, timeout=3)).click()

        await (await tab.select(self.selectors.login.email_input)).send_keys(self.email)
        await (await tab.select(self.selectors.login.password_input)).send_keys(self.password)
        await (await tab.select(self.selectors.login.login_button)).click()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + 20
        while loop.time() < deadline:
            if await self._exists(self.selectors.interactions.textbox):
                self.logger.info("Successfully logged into DeepSeek.")
                return
            body = await self._js("document.body.innerText") or ""
            if self.selectors.login.verification_text in body:
                raise VerificationRequired(
                    "DeepSeek requires email verification for this login. Use a token instead, or log in once "
                    "with headless=False and a persistent user_data_dir."
                )
            await asyncio.sleep(0.5)
        raise InvalidCredentials("Login failed, check your credentials.")

    async def send_message(
        self,
        message: str,
        slow_mode: bool = False,
        deepthink: bool = False,
        search: bool = False,
        timeout: int = 60,
        slow_mode_delay: float = 0.25,
        chat_id: str | None = None,
        system_prompt: str | None = None,
        files: list[str] | None = None,
    ) -> Response | None:
        """
        Sends a message to DeepSeek and retrieves the response.

        Args:
            message (str): Message text to send.
            slow_mode (bool): Send message slowly, char by char.
            deepthink (bool): Activate DeepThink feature.
            search (bool): Activate search feature.
            timeout (int): Max wait time for response.
            slow_mode_delay (float): Delay between chars in slow mode.
            chat_id (str | None): ID of the chat to switch to.
            system_prompt (str | None): Instructions prepended to the message in the request payload only, so
                they don't appear in the chat box.
            files (list[str] | None): Paths of files (images, documents, code...) to attach to the message.

        Returns:
            Optional[Response]: The DeepSeek response or None if timed out.
        """
        response = None
        async for event in self.stream_message(
            message,
            slow_mode=slow_mode,
            deepthink=deepthink,
            search=search,
            timeout=timeout,
            slow_mode_delay=slow_mode_delay,
            chat_id=chat_id,
            system_prompt=system_prompt,
            files=files,
        ):
            if event.type == "done":
                response = event.response
        return response

    async def stream_message(
        self,
        message: str,
        slow_mode: bool = False,
        deepthink: bool = False,
        search: bool = False,
        timeout: int = 60,
        slow_mode_delay: float = 0.25,
        chat_id: str | None = None,
        system_prompt: str | None = None,
        files: list[str] | None = None,
        new_chat: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """
        Sends a message to DeepSeek and yields the response as it streams.

        Takes the same arguments as `send_message`, plus `new_chat` to start a new chat first. Starting it here, under
        the same lock as the send, keeps a concurrent request from switching chats in between.

        Yields "thinking" and "text" deltas, then a final "done" event carrying the complete `Response`. The timeout includes queueing,
        recovery, browser setup and response generation.

        Raises:
            FileUploadError: If an attached file is missing, or DeepSeek fails to process it.
        """
        if not self._initialized:
            raise MissingInitialization("Call `initialize()` before sending messages.")

        deadline = asyncio.get_running_loop().time() + timeout
        acquired = False
        started = False
        completed = False
        rejected = False
        stream = None
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout)
            acquired = True
            remaining = deadline - asyncio.get_running_loop().time()
            await asyncio.wait_for(self._ensure_browser(), max(0, remaining))
            started = True
            stream = self._stream_message_locked(
                message, slow_mode, deepthink, search, timeout, slow_mode_delay,
                chat_id, system_prompt, files, new_chat,
            )
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    event = await asyncio.wait_for(anext(stream), max(0, remaining))
                except StopAsyncIteration:
                    return
                if event.type == "done":
                    completed = True
                yield event
        except RateLimited:
            rejected = True
            raise
        except asyncio.TimeoutError as exc:
            raise ServerDown("DeepSeek request deadline exceeded (queue or browser operation)") from exc
        except Exception as exc:
            raise ServerDown(f"DeepSeek browser operation failed: {exc}") from exc
        finally:
            if acquired:
                try:
                    if stream is not None:
                        await stream.aclose()
                finally:
                    if started and not completed and not rejected:
                        self._schedule_recovery()
                    self._lock.release()

    async def _stream_message_locked(
        self, message, slow_mode, deepthink, search, timeout, slow_mode_delay,
        chat_id, system_prompt, files, new_chat,
    ):
        tab = self.browser.main_tab

        if new_chat:
            await self._open_new_chat()
        elif chat_id is not None and chat_id != self._current_chat_id():
            self.logger.debug(f"Switching to chat ID: {chat_id}")
            await self._browser_call(tab.get(f"{BASE_URL}a/chat/s/{chat_id}"))

        # Wait for the chat to load
        await self._browser_call(tab.wait_for(self.selectors.interactions.textbox, timeout=10))
        if files:
            await self._attach_files(files, timeout)
        textbox = await self._browser_call(tab.select(self.selectors.interactions.textbox))
        await self._browser_call(textbox.click())

        self.logger.debug(f"Sending message: {message}")
        # insertText avoids key events, so newlines in the message don't trigger an early send
        if slow_mode:
            for char in message:
                await asyncio.sleep(slow_mode_delay)
                await self._browser_call(tab.send(cdp.input_.insert_text(char)))
                await asyncio.sleep(slow_mode_delay)
        else:
            # DeepSeek's editor can take a while to take in long text, such as several tool results at once
            await self._browser_call(tab.send(cdp.input_.insert_text(message)), extra=len(message) / 1000)

        # Arm the interceptor: the next completion request gets these payload changes
        token = uuid.uuid4().hex
        prefix = f"<system_instructions>\n{system_prompt}\n</system_instructions>\n\n" if system_prompt else ""
        pending = {"token": token, "prefix": prefix, "thinking": deepthink, "search": search}
        self._events = asyncio.Queue()
        self._active_token = token
        await self._js(f"window.__openseek.pending = {json.dumps(pending)}")

        button = await self._browser_call(tab.select(self.selectors.interactions.send_button))
        await self._browser_call(button.click())
        try:
            async for event in self._read_stream(timeout, expected_files=len(files or [])):
                yield event
        finally:
            self._active_token = None

    async def _attach_files(self, files: list[str], timeout: float) -> None:
        """Attaches files through the page's file input and waits until DeepSeek has processed them."""
        paths = [Path(path).resolve() for path in files]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileUploadError(f"File not found: {', '.join(missing)}")

        file_input = await self._browser_call(
            self.browser.main_tab.select(self.selectors.interactions.file_input)
        )
        # The page silently drops files whose extension isn't in the input's `accept` list
        accepted = {ext.strip().lower() for ext in (file_input.attrs.get("accept") or "").split(",") if ext.strip()}
        unsupported = [path.name for path in paths if accepted and path.suffix.lower() not in accepted]
        if unsupported:
            raise FileUploadError(f"DeepSeek does not support these file types: {', '.join(unsupported)}")

        self.logger.debug(f"Attaching files: {paths}")
        await self._js("window.__openseek.files = {}; window.__openseek.fileErrors = []")
        await self._browser_call(file_input.send_file(*paths))

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            await asyncio.sleep(0.5)
            state = json.loads(await self._js("JSON.stringify(window.__openseek)"))
            uploaded = list(state["files"].values())
            failed = [f for f in uploaded if f.get("status") in FAILED_FILE_STATUSES or f.get("error_code")]
            if state["fileErrors"] or failed:
                await self._discard_attachments()
                details = state["fileErrors"] + [
                    f"{f.get('file_name')}: {f.get('error_code') or f.get('status')}" for f in failed
                ]
                raise FileUploadError(f"DeepSeek could not process the attached files: {'; '.join(details)}")
            if len(uploaded) >= len(paths) and all(f.get("status") == "SUCCESS" for f in uploaded):
                self.logger.debug(f"Files ready: {[f.get('id') for f in uploaded]}")
                return

        await self._discard_attachments()
        raise FileUploadError("Timed out waiting for DeepSeek to process the attached files.")

    async def _discard_attachments(self) -> None:
        """Reloads the chat so attachments that failed don't stay in the composer for the next message."""
        tab = self.browser.main_tab
        await self._browser_call(tab.get(tab.url or BASE_URL))
        with contextlib.suppress(asyncio.TimeoutError):
            await tab.wait_for(self.selectors.interactions.textbox, timeout=15)

    async def _read_stream(self, timeout: int, expected_files: int = 0) -> AsyncIterator[StreamEvent]:
        """Turns interceptor events for the armed request into stream events."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        sse, state = SSEParser(), ResponseState()
        chat_id, raw = None, ""

        while True:
            remaining = deadline - loop.time()
            try:
                if remaining <= 0:
                    raise asyncio.TimeoutError
                message = await asyncio.wait_for(self._events.get(), timeout=remaining)
            except asyncio.TimeoutError:
                if chat_id is None:
                    self.logger.error("The chat request was not intercepted, the page may have changed.")
                else:
                    self.logger.error("Timed out waiting for response.")
                return

            kind = message.get("kind")
            if kind == "request":
                chat_id = message.get("chat_id")
                self.active_chat_id = chat_id or self._current_chat_id()
                attached = len(message.get("ref_file_ids") or [])
                if attached < expected_files:
                    self.logger.warning(f"Only {attached} of {expected_files} files were attached to the request.")
            elif kind == "error":
                self.logger.warning(f"Could not modify the chat request: {message.get('error')}")
            elif kind == "transport_error":
                raise ServerDown(f"DeepSeek connection interrupted: {message.get('error')}")
            elif kind == "chunk":
                raw += message["data"]
                for event, data in sse.feed(message["data"]):
                    for fragment_type, text in state.feed(event, data):
                        if fragment_type == THINK:
                            yield StreamEvent("thinking", text)
                        elif fragment_type == RESPONSE:
                            yield StreamEvent("text", text)
            elif kind == "done":
                yield StreamEvent("done", response=self._build_response(state, raw, message.get("status"), chat_id))
                return

    def _build_response(self, state: ResponseState, raw: str, status: int | None, chat_id: str | None) -> Response:
        detail = state.error
        if not detail and not state.fragments:
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                detail = json.loads(raw).get("msg")
        if not state.fragments and (
            status == 429 or (isinstance(detail, str) and "messages too frequent" in detail.lower())
        ):
            raise RateLimited(f"DeepSeek rate limited the request: {detail or 'HTTP 429'}. Try again later.")
        if status is None or not 200 <= status < 300:
            raise ServerDown(f"DeepSeek request failed (HTTP {status})")
        if state.error:
            raise ServerDown(f"DeepSeek returned an error: {state.error}")
        if not state.fragments:
            # Errors come back as a plain JSON body instead of an event stream
            detail = raw.strip()
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                detail = json.loads(detail).get("msg") or detail
            raise ServerDown(f"DeepSeek returned no response (HTTP {status}): {detail[:200] or 'empty body'}")

        text = state.text_of(RESPONSE).strip()
        if "server is busy" in text.lower():
            raise ServerDown("Server busy, try again later.")

        think = next((f for f in state.fragments if f.get("type") == THINK), None)
        search_results = [
            result for f in state.fragments for result in (f.get("results") or f.get("references") or [])
        ]

        return Response(
            text=text,
            chat_id=chat_id or self._current_chat_id(),
            deepthink_duration=think.get("elapsed_secs") if think else None,
            deepthink_content=state.text_of(THINK).strip() or None,
            search_results=search_results or None,
            message_id=state.message_id,
            token_usage=state.response.get("accumulated_token_usage"),
        )

    def _current_chat_id(self) -> str | None:
        url = self.browser.main_tab.url or ""
        return url.rstrip("/").split("/")[-1] if "/a/chat/s/" in url else None

    async def start_new_chat(self) -> None:
        """
        Starts a new chat session.
        """
        if not self._initialized:
            raise MissingInitialization("Call `initialize()` before starting a new chat.")

        await asyncio.wait_for(self._start_new_chat_locked(), self.recovery_timeout + self.browser_timeout)

    async def _start_new_chat_locked(self):
        async with self._lock:
            await self._ensure_browser()
            try:
                await self._open_new_chat()
            except (Exception, asyncio.CancelledError):
                self._schedule_recovery()
                raise

    async def _open_new_chat(self) -> None:
        tab = self.browser.main_tab
        await self._browser_call(tab.get(BASE_URL))
        await self._browser_call(tab.wait_for(self.selectors.interactions.textbox, timeout=15))
        self.logger.debug("New chat started.")

    async def close(self) -> None:
        """
        Cleans up resources and closes browser.
        """
        self._initialized = False
        self._healthy = False
        for task in (self._keepalive_task, self._recovery_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await self._stop_browser()
        if hasattr(self, "display"):
            self.display.stop()

    @staticmethod
    def run_sync(coro: asyncio.coroutines) -> any:
        """Runs asynchronous coroutine synchronously."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def send_message_sync(self, *args, **kwargs) -> Response | None:
        """Synchronous wrapper for `send_message`."""
        return self.run_sync(self.send_message(*args, **kwargs))

    def start_new_chat_sync(self) -> None:
        """
        Synchronous wrapper for `start_new_chat`.
        """
        self.run_sync(self.start_new_chat())

    def initialize_sync(self):
        """Synchronous wrapper for `initialize`."""
        self.run_sync(self.initialize())

    def close_sync(self):
        """
        Synchronous wrapper for `close`.
        """
        self.run_sync(self.close())
