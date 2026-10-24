import json
import unittest
from collections import OrderedDict
from unittest.mock import Mock, patch

import server
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.tools import RETRY_NOTE, has_malformed_calls

NAMES = {'read', 'edit'}
TOOLS = [{'type': 'function', 'function': {'name': name}} for name in sorted(NAMES)]
# From a transcript: a duplicated parameter and an invoke that lost its keyword, which can't be parsed safely
GARBLED = '''<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/exposure.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="limit" string="false">120</｜｜DSML｜｜>
<｜｜DSML｜｜ parameter name="path" string="true">/x/exposure.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ name="read">
<｜｜DSML｜｜ parameter name="offset" string="false">458</｜｜DSML｜｜>
<｜｜DSML｜｜ calls>'''
FIRST = 'Checking the code.\n\n' + GARBLED
CALL = '<tool_call>\n{"name": "read", "arguments": {"path": "/x/exposure.py", "offset": 458}}\n</tool_call>'


class DetectionTests(unittest.TestCase):
    def test_malformed_markup_outside_fences(self):
        self.assertTrue(has_malformed_calls(FIRST, NAMES))
        self.assertTrue(has_malformed_calls('<tool_call>{"name": "read", "arguments": {}', NAMES))
        # A well-formed call to a tool the request doesn't have is a failed call too
        self.assertTrue(has_malformed_calls('<tool_call>{"name": "shell", "arguments": {}}</tool_call>', NAMES))
        for text in ['Done.', CALL, 'Example:\n```\n' + GARBLED + '\n```\nDone.', FIRST + '\n' + CALL]:
            self.assertFalse(has_malformed_calls(text, NAMES), text)


class RetryTests(unittest.IsolatedAsyncioTestCase):
    def fake(self, *replies):
        self.sent, self.closed = [], []

        async def events(message, **kwargs):
            self.sent.append((message, kwargs))
            text = replies[len(self.sent) - 1]
            try:
                for char in text:
                    yield StreamEvent('text', char)
                yield StreamEvent('done', response=Response(text=text, chat_id=kwargs.get('chat_id') or 'chat'))
            finally:
                self.closed.append(True)

        ds = Mock()
        ds.stream_message = events
        ds.active_chat_id = 'chat'
        return ds

    async def complete(self, ds, streaming, tools=TOOLS, sessions=None):
        sessions = OrderedDict() if sessions is None else sessions
        with patch.object(server.state, 'deepseek', ds), patch.object(server.state, 'sessions', sessions), \
                patch.object(server.state, 'chats', OrderedDict()), patch.object(server, '_save_sessions'):
            request = server._ChatRequest(messages=[{'role': 'user', 'content': 'go'}], stream=streaming, tools=tools)
            if not streaming:
                message = (await server.chat_completions(request, 's'))['choices'][0]['message']
                return message['content'] or '', message.get('tool_calls', [])
            response = await server.chat_completions(request, 's')
            payloads = [json.loads(p[6:]) for p in [part async for part in response.body_iterator]
                        if p.startswith('data: {')]
            deltas = [p['choices'][0]['delta'] for p in payloads]
            content = ''.join(d.get('content', '') for d in deltas)
            return content, [c for d in deltas for c in d.get('tool_calls', [])]

    async def test_malformed_call_is_written_again_in_the_same_chat(self):
        for streaming in [False, True]:
            ds = self.fake(FIRST, CALL)
            sessions = OrderedDict()
            content, calls = await self.complete(ds, streaming, sessions=sessions)
            # The malformed markup is never shown; the text before it is kept
            self.assertEqual(content.strip(), 'Checking the code.')
            self.assertEqual([json.loads(c['function']['arguments']) for c in calls],
                             [{'path': '/x/exposure.py', 'offset': 458}])
            message, kwargs = self.sent[1]
            self.assertEqual(message, RETRY_NOTE)
            self.assertEqual(kwargs['chat_id'], 'chat')
            self.assertFalse(kwargs.get('new_chat'))
            self.assertEqual(self.closed, [True, True])
            # The session continues the chat, keyed on the reply the client received
            self.assertEqual(sessions['s'].chat_id, 'chat')
            reply = server._message_key('assistant', content, calls)
            self.assertEqual(sessions['s'].keys[-1], reply)

    async def test_retries_only_once(self):
        for streaming in [False, True]:
            ds = self.fake(FIRST, GARBLED)
            content, calls = await self.complete(ds, streaming)
            self.assertEqual(len(self.sent), 2)
            self.assertEqual(calls, [])
            self.assertEqual(content.strip(), 'Checking the code.\n\n' + GARBLED)

    async def test_no_retry_without_a_failed_call(self):
        fenced = 'Example:\n```\n' + GARBLED + '\n```\nDone.'
        for streaming in [False, True]:
            for text in ['Done.', 'Reading.\n' + CALL, fenced]:
                ds = self.fake(text)
                content, _ = await self.complete(ds, streaming)
                self.assertEqual(len(self.sent), 1)
                if text == fenced:
                    self.assertEqual(content, fenced)
            ds = self.fake(FIRST)
            content, _ = await self.complete(ds, streaming, tools=None)
            self.assertEqual((len(self.sent), content), (1, FIRST))
            with patch.object(server, 'RETRY_MALFORMED_CALLS', False):
                ds = self.fake(FIRST)
                content, _ = await self.complete(ds, streaming)
            self.assertEqual((len(self.sent), content), (1, FIRST))

    async def test_text_after_a_valid_call_streams_without_waiting(self):
        released = []

        async def events():
            for part in ['Reading.\n', CALL, '\nMore']:
                yield StreamEvent('text', part)
            released.append(True)
            yield StreamEvent('done', response=Response(text='Reading.\n' + CALL + '\nMore', chat_id='chat'))

        stream = server._retry_malformed_calls(events(), NAMES, None)
        parts = []
        async for event in stream:
            if event.type == 'text':
                parts.append((event.content, bool(released)))
        self.assertEqual(parts[-1], ('\nMore', False))
        self.assertEqual(''.join(part for part, _ in parts), 'Reading.\n' + CALL + '\nMore')


if __name__ == '__main__':
    unittest.main()
