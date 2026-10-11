import json
import unittest
from collections import OrderedDict

import server
from openseek import DeepSeek
from openseek.internal.exceptions import ServerDown
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.stream import ResponseState


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.previous = server.state.sessions
        server.state.sessions = OrderedDict()

    def tearDown(self):
        server.state.sessions = self.previous

    def test_significant_edits_change_keys(self):
        key = server._message_key
        self.assertNotEqual(key('user', 'print("a  b")'), key('user', 'print("a b")'))
        def call(name, arg):
            return [{"id": "same", "type": "function", "function": {"name": name, "arguments": arg}}]
        self.assertNotEqual(key('assistant', None, call('read', 'a')),
                            key('assistant', None, call('read', 'b')))
        self.assertNotEqual(key('assistant', None, call('read', 'a')),
                            key('assistant', None, call('write', 'a')))
        self.assertEqual(key('assistant', None, call('read', '{"a": 1, "b": 2}')),
                         key('assistant', None, call('read', '{"b":2,"a":1}')))
        self.assertNotEqual(key('assistant', None, call('read', '{"a":"x  y"}')),
                            key('assistant', None, call('read', '{"a":"x y"}')))

    def test_reuse_edit_and_compaction(self):
        messages = [server._Message(role='user', content='a  b'),
                    server._Message(role='assistant', content='OK'),
                    server._Message(role='user', content='continue')]
        keys = lambda: [server._message_key(m.role, m.content) for m in messages]
        server.state.sessions['s'] = server._Chat(chat_id='chat', instructions_hash=None, keys=keys()[:2])
        known, pending, compacting = server._match_session('s', messages, keys(), 2)
        self.assertEqual(known.chat_id, 'chat')
        self.assertEqual(pending, messages[2:])
        self.assertFalse(compacting)
        messages[0].content = 'a b'
        self.assertIsNone(server._match_session('s', messages, keys(), 2)[0])
        messages[0].content = 'a  b'
        messages[2].content = server.COMPACTION_PROMPT_PREFIXES[0]
        self.assertTrue(server._match_session('s', messages, keys(), 2)[2])


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, text, tools=True):
        async def events():
            for char in text:
                yield StreamEvent('text', char)
            yield StreamEvent('done', response=Response(text=text.strip(), chat_id='chat'))

        remembered = []
        chunks = [part async for part in server._stream_completion(
            events(), 'id', 0, 'deepseek-chat', tools, lambda *args: remembered.append(args))]
        payloads = [json.loads(part[6:]) for part in chunks if part.startswith('data: {')]
        content = ''.join(p['choices'][0]['delta'].get('content', '') for p in payloads)
        return content, remembered, payloads

    async def test_invalid_and_partial_blocks_keep_position(self):
        for text in ['Before <tool_call>not JSON</tool_call> after',
                     'Before <tool_call>unfinished',
                     'Before <｜DSML｜function_calls>invalid</｜DSML｜function_calls> after']:
            content, remembered, _ = await self.collect(text)
            self.assertEqual(content, text)
            self.assertEqual(remembered[0][1], text)

    async def test_valid_tools_and_exact_client_history(self):
        text = '  Before\n\n\n<tool_call>{"name":"read","arguments":{}}</tool_call>\n after  '
        content, remembered, payloads = await self.collect(text)
        self.assertEqual(content, '  Before\n\n\n\n after  ')
        self.assertEqual(remembered[0][1], content)
        self.assertEqual(len(remembered[0][2]), 1)
        self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'tool_calls')
        content, remembered, _ = await self.collect('  exact\n\n text  ', tools=False)
        self.assertEqual(remembered[0][1], content)

    async def test_misspelled_and_missing_close_tags(self):
        read = '{"name": "read", "arguments": {"path": "%s"}}'
        text = ('Reading.\n<tool_call>\n' + read % 'a' + '\n</call_call>\n<tool_call>\n' + read % 'b'
                + '\n</tool_call>\n<tool_call>\n' + read % 'c' + '\n<tool_call>\n' + read % 'd' + '\n</call_call>')
        content, remembered, payloads = await self.collect(text)
        self.assertEqual(content.strip(), 'Reading.')
        paths = [json.loads(call['function']['arguments'])['path'] for call in remembered[0][2]]
        self.assertEqual(paths, ['a', 'b', 'c', 'd'])
        self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'tool_calls')

    async def test_dsml_without_inner_delimiters(self):
        text = ('Searching.\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="grep">\n'
                '<｜｜DSML｜｜ parameter name="pattern" string="true">def |class </｜｜DSML｜｜ parameter>\n'
                '<｜｜DSML｜｜ parameter name="limit" string="false">150</parameter>\n'
                '</invoke>\n</｜｜DSML｜｜ calls>')
        content, remembered, _ = await self.collect(text)
        self.assertEqual(content.strip(), 'Searching.')
        call = remembered[0][2][0]['function']
        self.assertEqual(call['name'], 'grep')
        self.assertEqual(json.loads(call['arguments']), {'pattern': 'def |class ', 'limit': 150})

    async def test_transport_failure_never_remembers_or_finishes_successfully(self):
        ds = DeepSeek(token='test')
        ds._events.put_nowait({'kind': 'chunk', 'data': 'data: ' + json.dumps({
            'v': {'response': {'fragments': [{'type': 'RESPONSE', 'content': 'partial'}]}}
        }) + '\n\n'})
        ds._events.put_nowait({'kind': 'transport_error', 'error': 'connection reset'})
        remembered = []
        chunks = [part async for part in server._stream_completion(
            ds._read_stream(1), 'id', 0, 'deepseek-chat', False, lambda *args: remembered.append(args))]
        self.assertTrue(any('connection reset' in chunk for chunk in chunks))
        self.assertFalse(any('"finish_reason": "stop"' in chunk for chunk in chunks))
        self.assertEqual(remembered, [])

    async def test_failed_status_rejects_partial_response(self):
        ds = DeepSeek(token='test')
        state = ResponseState()
        state.root = {'response': {'fragments': [{'type': 'RESPONSE', 'content': 'partial'}]}}
        for status in [None, 0, 403, 500]:
            with self.assertRaises(ServerDown):
                ds._build_response(state, '', status, 'chat')
        self.assertEqual(ds._build_response(state, '', 200, 'chat').text, 'partial')


if __name__ == '__main__':
    unittest.main()
