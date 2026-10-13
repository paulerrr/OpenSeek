import json
import unittest
from collections import OrderedDict
from unittest.mock import AsyncMock, Mock, patch

import server
from openseek import DeepSeek
from openseek.internal.exceptions import ServerDown
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.tools import ToolOutputGuard, parse_tool_calls

TAG = '<｜｜DSML｜｜ parameter>'


class GuardTests(unittest.TestCase):
    def test_repetition_is_chunk_independent(self):
        for delimiter in ['', '\n', ' \n']:
            text = delimiter.join([TAG] * 40)
            for size in [1, 2, 7, 53, len(text)]:
                guard = ToolOutputGuard()
                emitted = ''
                with self.assertRaises(ServerDown):
                    for start in range(0, len(text), size):
                        emitted += guard.feed(text[start:start + size])
                self.assertNotIn('DSML', emitted)

    def test_nonrunaway_text_is_preserved(self):
        for text in [TAG, TAG * 3, 'Before\n' + TAG + '\nafter',
                     '```xml\n' + (TAG + '\n') * 10 + '```\n',
                     '~~~xml\n' + (TAG + '\n') * 10 + '~~~\n',
                     '<｜DSML｜invoke name="read"><｜DSML｜parameter name="x">1</parameter></invoke>']:
            for size in [1, 7, len(text)]:
                guard = ToolOutputGuard()
                output = ''.join(guard.feed(text[i:i + size]) for i in range(0, len(text), size))
                self.assertEqual(output + guard.finish(), text)

    def test_malformed_invocation_never_becomes_empty_call(self):
        for body in [TAG, '<｜DSML｜parameter name="path">unfinished',
                     '<｜DSML｜parameter name="path">a</parameter>' * 2]:
            text = '<｜DSML｜invoke name="read">' + body + '</invoke>'
            self.assertEqual(parse_tool_calls(text, {'read'}), (text, []))
        self.assertEqual(len(parse_tool_calls('<｜DSML｜invoke name="read"></invoke>', {'read'})[1]), 1)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_errors_and_forgets_failed_session(self):
        for streaming in [False, True]:
            closed = []

            async def events(*args, **kwargs):
                try:
                    for char in (TAG + '\n') * 40:
                        yield StreamEvent('text', char)
                    yield StreamEvent('done', response=Response(text='unreachable', chat_id='bad'))
                finally:
                    closed.append(True)

            ds = Mock()
            ds.stream_message = events
            request = server._ChatRequest(
                messages=[{'role': 'user', 'content': 'go'}], stream=streaming,
                tools=[{'type': 'function', 'function': {'name': 'read'}}])
            sessions = OrderedDict(s=server._Chat(chat_id='old', instructions_hash=None))
            with patch.object(server.state, 'deepseek', ds), patch.object(server.state, 'sessions', sessions), \
                    patch.object(server.state, 'chats', OrderedDict()), patch.object(server, '_save_sessions'):
                if streaming:
                    response = await server.chat_completions(request, 's')
                    chunks = [part async for part in response.body_iterator]
                    payloads = [json.loads(p[6:]) for p in chunks if p.startswith('data: {')]
                    self.assertEqual(payloads[-1]['error']['code'], 503)
                    self.assertFalse(any(p.get('choices', [{}])[0].get('finish_reason') for p in payloads))
                    self.assertFalse(any('DSML' in p.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                         for p in payloads))
                else:
                    with self.assertRaises(server.HTTPException) as error:
                        await server.chat_completions(request, 's')
                    self.assertEqual(error.exception.status_code, 503)
                self.assertNotIn('s', sessions)
                self.assertFalse(server.state.chats)
                self.assertEqual(closed, [True])

    async def test_aborted_browser_turn_reloads_before_unlocking(self):
        ds = DeepSeek(token='test')
        ds._initialized = True
        tab = Mock()
        tab.wait_for = AsyncMock()
        tab.send = AsyncMock()
        tab.select = AsyncMock(return_value=Mock(click=AsyncMock()))
        tab.reload = AsyncMock(side_effect=lambda: self.assertTrue(ds._lock.locked()))
        ds.browser = Mock(main_tab=tab)
        ds._js = AsyncMock()

        async def events(*args, **kwargs):
            yield StreamEvent('text', TAG * 4)

        ds._read_stream = events
        with self.assertRaises(ServerDown):
            async for _ in server._guard_tool_output(ds.stream_message('go')):
                pass
        tab.reload.assert_awaited_once()
        self.assertFalse(ds._lock.locked())
        self.assertIsNone(ds._active_token)
