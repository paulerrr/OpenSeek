"""Reduced fixtures from the GPU-search conversation; no external calls are executed."""

import json
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import server
from openseek import DeepSeek
from openseek.internal.exceptions import RateLimited, ServerDown
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.stream import ResponseState
from openseek.internal.tools import ToolCallStreamFilter, parse_tool_calls


OPEN = '<｜｜DSML｜｜ calls>'
CLOSE = '</｜｜DSML｜｜ calls>'
JSON_CALL = '<tool_call>{"name":"shell","arguments":{"command":"echo test"}}</tool_call>'
INVOKE = '''<｜｜DSML｜｜ invoke name="grep">
<｜｜DSML｜｜ pattern" string="true">low.?profile|half.?height</｜｜DSML｜｜ pattern>
<｜｜DSML｜｜ parameter name="path" string="true">/project</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>'''


class ConversationParserTests(unittest.TestCase):
    def test_wrappers_and_named_parameters(self):
        fixtures = [
            (INVOKE, ['grep']),
            (INVOKE.replace('</｜｜DSML｜｜ invoke>', '<｜｜DSML｜｜ invoke>'), ['grep']),
            (JSON_CALL, ['shell']),
            (JSON_CALL + '\n</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>', ['shell']),
            (JSON_CALL + '\n' + INVOKE, ['shell', 'grep']),
            (JSON_CALL.replace('</tool_call>', '</call>'), ['shell']),
        ]
        for body, expected in fixtures:
            text = 'Checking.\n' + OPEN + '\n' + body + '\n' + CLOSE + '\nAfter.'
            with self.subTest(body=body):
                content, calls = parse_tool_calls(text, {'shell', 'grep'})
                self.assertEqual([c['function']['name'] for c in calls], expected)
                self.assertEqual(content, 'Checking.\n\nAfter.')
                for call in calls:
                    args = json.loads(call['function']['arguments'])
                    if call['function']['name'] == 'grep':
                        self.assertEqual(args, {'pattern': 'low.?profile|half.?height', 'path': '/project'})
                    else:
                        self.assertEqual(args, {'command': 'echo test'})
                for size in (1, 2, 7, 31, len(text)):
                    filt = ToolCallStreamFilter({'shell', 'grep'})
                    emitted = ''.join(filt.feed(text[i:i + size]) for i in range(0, len(text), size))
                    self.assertEqual((emitted + filt.finish()).strip(), content)

    def test_ambiguous_wrappers_never_execute_partial_calls(self):
        for body in [
            JSON_CALL + '\nunexpected content',
            JSON_CALL + INVOKE.replace('</｜｜DSML｜｜ pattern>', '</｜｜DSML｜｜ other>'),
            JSON_CALL + INVOKE.replace('name="grep"', 'name="unknown"'),
            JSON_CALL + INVOKE.replace('name="path"', 'name="pattern"'),
            JSON_CALL + '<tool_call>{invalid}</tool_call>',
        ]:
            text = OPEN + body + CLOSE
            with self.subTest(body=body):
                self.assertEqual(parse_tool_calls(text, {'shell', 'grep'}), (text, []))
                filt = ToolCallStreamFilter({'shell', 'grep'})
                self.assertEqual(''.join(filt.feed(c) for c in text) + filt.finish(), text)


class RateLimitTests(unittest.IsolatedAsyncioTestCase):
    def test_classifies_explicit_refusals_only(self):
        ds = DeepSeek(token='test')
        for status, raw, error in [
            (429, '', None),
            (200, '{"msg":"Messages too frequent. Try again later."}', None),
            (200, '', 'Messages too frequent. Try again later.'),
        ]:
            state = ResponseState()
            state.error = error
            with self.assertRaises(RateLimited):
                ds._build_response(state, raw, status, 'chat')
        for state, status in [(ResponseState(), 503), (ResponseState(), 200)]:
            with self.assertRaises(ServerDown) as caught:
                ds._build_response(state, '', status, 'chat')
            self.assertNotIsInstance(caught.exception, RateLimited)
        state = ResponseState()
        state.root['response']['fragments'] = [{'type': 'RESPONSE', 'content': 'partial'}]
        state.error = 'Messages too frequent'
        with self.assertRaises(ServerDown) as caught:
            ds._build_response(state, '', 200, 'chat')
        self.assertNotIsInstance(caught.exception, RateLimited)

    async def test_refusal_preserves_browser_and_releases_lock(self):
        ds = DeepSeek(token='test')
        ds._initialized = ds._healthy = True
        ds._ensure_browser = AsyncMock()
        ds._schedule_recovery = Mock()

        async def refused(*args, **kwargs):
            raise RateLimited('Messages too frequent')
            yield

        ds._stream_message_locked = refused
        with self.assertRaises(RateLimited):
            _ = [e async for e in ds.stream_message('hello')]
        self.assertFalse(ds._lock.locked())
        ds._schedule_recovery.assert_not_called()

        async def success(*args, **kwargs):
            yield StreamEvent('done', response=Response(text='OK', chat_id='chat'))

        ds._stream_message_locked = success
        result = [e async for e in ds.stream_message('hello')]
        self.assertEqual(result[0].response.text, 'OK')

    async def test_api_refusal_preserves_session_and_reports_429(self):
        async def refused(*args, **kwargs):
            raise RateLimited('Messages too frequent')
            yield

        for streaming in (False, True):
            for session_id in (None, 'session'):
                messages = [server._Message(role='user', content='first'),
                            server._Message(role='assistant', content='reply'),
                            server._Message(role='user', content='next')]
                keys = [server._message_key(m.role, m.content) for m in messages[:2]]
                known = server._Chat(chat_id='chat', instructions_hash=None, keys=keys)
                key = server._conversation_key(keys)
                chats = OrderedDict({key: known})
                sessions = OrderedDict({'session': known})
                fake = SimpleNamespace(stream_message=refused, active_chat_id='chat')
                req = server._ChatRequest(messages=messages, stream=streaming,
                                          tools=[{'function': {'name': 'shell', 'parameters': {}}}])
                with self.subTest(stream=streaming, session=session_id), \
                        patch.object(server.state, 'deepseek', fake), \
                        patch.object(server.state, 'chats', chats), \
                        patch.object(server.state, 'sessions', sessions), \
                        patch.object(server, '_save_sessions'):
                    if streaming:
                        response = await server.chat_completions(req, session_id)
                        chunks = [s async for s in response.body_iterator]
                        payloads = [json.loads(s[6:]) for s in chunks if s.startswith('data: {')]
                        error = next(p['error'] for p in payloads if 'error' in p)
                        self.assertEqual(error['code'], 429)
                        self.assertEqual(error['retry_after'], 10)
                        self.assertFalse(any(p.get('choices', [{}])[0].get('finish_reason') for p in payloads))
                    else:
                        with self.assertRaises(server.HTTPException) as caught:
                            await server.chat_completions(req, session_id)
                        self.assertEqual(caught.exception.status_code, 429)
                        self.assertEqual(caught.exception.headers['Retry-After'], '10')
                    self.assertIs(sessions['session'], known)
                    self.assertIs(chats[key], known)
