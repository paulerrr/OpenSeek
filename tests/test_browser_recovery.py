import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import server
from openseek import DeepSeek
from openseek.internal.exceptions import OpenSeekException, ServerDown
from openseek.internal.objects import Response, StreamEvent


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ds = DeepSeek(token='test')
        self.ds._initialized = self.ds._healthy = True
        self.ds.browser_timeout = 0.02
        self.ds._stop_browser = AsyncMock()
        async def launch():
            self.ds._healthy = True
        self.ds._launch_browser = AsyncMock(side_effect=launch)

    async def asyncTearDown(self):
        await self.ds.close()

    async def consume(self, **kwargs):
        return [event async for event in self.ds.stream_message('hello', **kwargs)]

    async def test_hung_browser_is_replaced_and_next_request_succeeds(self):
        tab = SimpleNamespace(wait_for=AsyncMock(side_effect=lambda *a, **k: None))
        async def hang(*args, **kwargs):
            await asyncio.Event().wait()
        tab.wait_for = hang
        self.ds.browser = SimpleNamespace(main_tab=tab)
        with self.assertRaises(ServerDown):
            await self.consume(timeout=1)
        await self.ds._recovery_task
        self.assertFalse(self.ds._lock.locked())
        self.ds._launch_browser.assert_awaited_once()
        async def success(*args):
            yield StreamEvent('done', response=Response(text='OK', chat_id='chat'))
        self.ds._stream_message_locked = success
        result = await self.consume(timeout=1)
        self.assertEqual(result[0].response.text, 'OK')

    async def test_queue_deadline_does_not_restart_active_browser(self):
        await self.ds._lock.acquire()
        try:
            with self.assertRaises(ServerDown):
                await self.consume(timeout=0.02)
            self.assertTrue(self.ds._lock.locked())
            self.assertIsNone(self.ds._recovery_task)
        finally:
            self.ds._lock.release()

    async def test_overall_deadline_applies_despite_continuous_output(self):
        async def endless(*args):
            while True:
                await asyncio.sleep(0.005)
                yield StreamEvent('text', 'x')
        self.ds._stream_message_locked = endless
        with self.assertRaises(ServerDown):
            await self.consume(timeout=0.03)
        await self.ds._recovery_task
        self.assertFalse(self.ds._lock.locked())

    async def test_cancellation_releases_lock_and_recovers(self):
        entered = asyncio.Event()
        async def hang(*args):
            entered.set()
            await asyncio.Event().wait()
            yield
        self.ds._stream_message_locked = hang
        task = asyncio.create_task(self.consume(timeout=1))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await self.ds._recovery_task
        self.assertFalse(self.ds._lock.locked())

    async def test_waiter_timeout_does_not_cancel_shared_recovery(self):
        ready = asyncio.Event()
        async def launch():
            await ready.wait()
            self.ds._healthy = True
        self.ds._launch_browser = AsyncMock(side_effect=launch)
        self.ds._schedule_recovery()
        with self.assertRaises(ServerDown):
            await self.consume(timeout=0.02)
        self.assertFalse(self.ds._recovery_task.done())
        ready.set()
        await self.ds._recovery_task
        self.assertTrue(self.ds._healthy)

    async def test_recovery_failure_can_retry(self):
        self.ds._launch_browser.side_effect = RuntimeError('startup failed')
        self.ds._schedule_recovery()
        await self.ds._recovery_task
        self.assertFalse(self.ds._healthy)
        async def launch():
            self.ds._healthy = True
        self.ds._launch_browser.side_effect = launch
        await self.ds._ensure_browser()
        self.assertTrue(self.ds._healthy)

    async def test_server_deadline_closes_stream(self):
        closed = asyncio.Event()
        async def hang():
            try:
                await asyncio.Event().wait()
                yield
            finally:
                closed.set()
        with self.assertRaises(OpenSeekException):
            async for _ in server._request_deadline(hang(), 0.02):
                pass
        self.assertTrue(closed.is_set())

    async def test_health_reports_recovery(self):
        previous = server.state.deepseek
        server.state.deepseek = self.ds
        try:
            self.ds._healthy = False
            with self.assertRaises(server.HTTPException) as error:
                await server.health()
            self.assertEqual(error.exception.status_code, 503)
        finally:
            server.state.deepseek = previous

    async def test_failed_browser_shutdown_kills_owned_process(self):
        process = Mock()
        process.poll.return_value = None
        browser = SimpleNamespace(stop=AsyncMock(side_effect=TimeoutError), _process=process)
        self.ds.browser = browser
        with patch("openseek.openseek.asyncio.to_thread", new=AsyncMock(side_effect=lambda fn: fn())):
            await DeepSeek._stop_browser(self.ds)
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    async def test_keepalive_timeout_schedules_recovery(self):
        self.ds.browser = SimpleNamespace(main_tab=SimpleNamespace(
            reload=AsyncMock(side_effect=TimeoutError)))
        async def stop_loop(*args):
            if self.ds._recovery_task is not None:
                raise asyncio.CancelledError
        with patch('openseek.openseek.asyncio.sleep', side_effect=stop_loop):
            with self.assertRaises(asyncio.CancelledError):
                await self.ds._keep_alive()
        await self.ds._recovery_task
        self.ds._launch_browser.assert_awaited_once()
