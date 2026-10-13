import json
import unittest
import unittest.mock
from collections import OrderedDict

import server
from openseek import DeepSeek
from openseek.internal import research
from openseek.internal.exceptions import ServerDown
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.research import parse_plan
from openseek.internal.stream import ResponseState
from openseek.internal.tools import ToolCallStreamFilter, parse_tool_calls


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


TOOL_NAMES = {'read', 'grep', 'shell', 'glob'}

MALFORMED_DSML = '''Let me look.
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/home/paul/projects/InstaFix</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="shell">
<｜｜DSML｜｜ parameter name="command" string="true">ls -la</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ invoke name="glob">
<｜｜DSML｜｜ parameter name="pattern" string="true">*/</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="limit" string="false">200</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''

MIXED_CALL = '''Reading.
<tool_call>
{"name": "read", "arguments": {"path": "/home/paul/projects/InstaFix/go.mod"}}
</｜｜DSML｜｜ parameter>
</tool_call>
</｜｜DSML｜｜ calls>'''

# A JSON call closed by a DSML invoke tag, followed by a DSML invoke with a plain </invoke>
JSON_THEN_DSML = '''Adding a style.
<tool_call>
{"name": "shell", "arguments": {"command": "ls"}}
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path">/x/styles.css</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="offset">30</｜｜DSML｜｜ parameter>
</invoke>
</｜｜DSML｜｜ calls>'''


APK_CALLS = [
    'Checking.\n<tool_call>\n'
    '{"name": "shell", "arguments": {"command": "python3 --version"}}\n'
    + closing + '\n<｜｜DSML｜｜ calls>'
    for closing in ['</call-tool>', '']
]
APK_CALLS.append(APK_CALLS[1].replace('<tool_call>\n', ''))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_malformed_calls(self):
        for text, names, prefix in [(MALFORMED_DSML, ['read', 'shell', 'glob'], 'Let me look.'),
                                    (MIXED_CALL, ['read'], 'Reading.'),
                                    (JSON_THEN_DSML, ['shell', 'read'], 'Adding a style.'),
                                    *((text, ['shell'], 'Checking.') for text in APK_CALLS)]:
            content, remembered, payloads = await self.collect(text)
            self.assertEqual(content.strip(), prefix)
            calls = remembered[0][2]
            self.assertEqual([c['function']['name'] for c in calls], names)
            self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'tool_calls')
            parsed_content, parsed_calls = parse_tool_calls(text, TOOL_NAMES)
            self.assertEqual(parsed_content, prefix)
            self.assertEqual([c['function'] for c in calls], [c['function'] for c in parsed_calls])
            for size in [1, 2, 7, 53, len(text)]:
                stream_filter = ToolCallStreamFilter(TOOL_NAMES)
                visible = ''.join(stream_filter.feed(text[i:i + size]) for i in range(0, len(text), size))
                visible += stream_filter.finish()
                self.assertEqual(visible.strip(), prefix)
        calls = parse_tool_calls(MALFORMED_DSML, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[2]['function']['arguments']), {'pattern': '*/', 'limit': 200})

    async def test_mixed_call_boundaries_preserve_following_content(self):
        call = '<tool_call>{"name":"shell","arguments":{"command":"ls"}}</call-tool>'
        cases = [
            (call + '\n<｜｜DSML｜｜ calls>unfinished', '<｜｜DSML｜｜ calls>unfinished', 1),
            (call + '\n<tool_call>{"name":"read","arguments":{}}</tool_call>', '', 2),
            (call + '\n<｜DSML｜calls><｜DSML｜invoke name="read"></invoke></｜DSML｜calls>', '', 2),
        ]
        for text, expected, count in cases:
            parsed, calls = parse_tool_calls(text, TOOL_NAMES)
            self.assertEqual(parsed, expected)
            self.assertEqual(len(calls), count)
            for size in [1, 7, len(text)]:
                filt = ToolCallStreamFilter(TOOL_NAMES)
                visible = ''.join(filt.feed(text[i:i + size]) for i in range(0, len(text), size)) + filt.finish()
                self.assertEqual(visible.strip(), expected)

    async def test_invalid_mixed_calls_remain_visible(self):
        for body in ['not JSON', '{"name":"unknown","arguments":{}}',
                     '{"name":"shell","arguments":42}']:
            text = '<tool_call>' + body + '</call-tool>\n<｜｜DSML｜｜ calls>'
            content, remembered, payloads = await self.collect(text)
            self.assertEqual(content, text)
            self.assertEqual(remembered[0][2], [])
            self.assertEqual(parse_tool_calls(text, TOOL_NAMES), (text, []))
            self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'stop')
        text = '<｜｜DSML｜｜ calls>'
        self.assertEqual((await self.collect(text))[0], text)

    async def test_unrelated_closing_markup_and_bad_json_are_preserved(self):
        for text in ['Example </｜｜DSML｜｜ calls>',
                     '<｜DSML｜calls></｜DSML｜calls>',
                     '<tool_call>{"name":"read"} unexpected</tool_call>',
                     '<｜｜DSML｜｜ invoke name="read">unfinished']:
            content, remembered, _ = await self.collect(text)
            self.assertEqual(content, text)
            self.assertEqual(remembered[0][2], [])

    async def collect(self, text, tools=True):
        async def events():
            for char in text:
                yield StreamEvent('text', char)
            yield StreamEvent('done', response=Response(text=text.strip(), chat_id='chat'))

        remembered = []
        chunks = [part async for part in server._stream_completion(
            events(), 'id', 0, 'deepseek-chat', TOOL_NAMES if tools else None,
            lambda *args: remembered.append(args))]
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

    async def test_bare_json_calls(self):
        shell = '{"name": "shell", "arguments": {"command": "echo %s"}}'
        text = ('Let me look.\n\n' + shell % 'a' + '\nPowerShell needs another form.\n\n  ' + shell % 'b'
                + '\n' + shell % 'c')
        content, remembered, payloads = await self.collect(text)
        self.assertEqual(content, 'Let me look.\n\n\nPowerShell needs another form.\n\n\n')
        commands = [json.loads(call['function']['arguments'])['command'] for call in remembered[0][2]]
        self.assertEqual(commands, ['echo a', 'echo b', 'echo c'])
        self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(parse_tool_calls(text, TOOL_NAMES)[0], 'Let me look.\n\nPowerShell needs another form.')

    async def test_bare_json_that_is_not_a_call_stays_text(self):
        for text in ['Example:\n```json\n{"name": "shell", "arguments": {}}\n```\nDone.',
                     '{"name": "delete_everything", "arguments": {}}',
                     'Inline {"name": "shell", "arguments": {}} mention',
                     '{"name": "shell", "arguments": {}} trailing words',
                     '{"name": "shell", "arguments": {"command": "unfinished',
                     '{"name": "shell"}',
                     '{"named": 1}\n{ not json }\n{',
                     '<tool_call>bad</tool_call> {"name": "shell", "arguments": {}}']:
            content, remembered, _ = await self.collect(text)
            self.assertEqual(content, text)
            self.assertEqual(remembered[0][2], [])
            self.assertEqual(parse_tool_calls(text, TOOL_NAMES)[1], [])
        content, remembered, _ = await self.collect('{"name": "shell", "arguments": {}}', tools=False)
        self.assertEqual(remembered[0][2], [])

    async def test_mixed_tagged_and_bare_calls_keep_order(self):
        text = ('<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>\n'
                '{"name": "grep", "arguments": {"pattern": "b"}}\n'
                '<tool_call>{"name": "read", "arguments": {"path": "c"}}</tool_call>')
        _, remembered, _ = await self.collect(text)
        self.assertEqual([call['function']['name'] for call in remembered[0][2]], ['read', 'grep', 'read'])
        self.assertEqual([call['function']['name'] for call in parse_tool_calls(text, TOOL_NAMES)[1]],
                         ['read', 'grep', 'read'])

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

    async def test_dsml_arguments_parameter_is_unwrapped(self):
        text = ('<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="shell">\n<｜｜DSML｜｜ parameter name="arguments" '
                'string="false">{"command": "uname -a"}</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
        _, remembered, _ = await self.collect(text)
        self.assertEqual(json.loads(remembered[0][2][0]['function']['arguments']), {'command': 'uname -a'})

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


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.previous = server.state.deepseek, server.state.chats
        server.state.chats = OrderedDict()

    def tearDown(self):
        server.state.deepseek, server.state.chats = self.previous

    async def complete(self, reply, tools):
        server.state.deepseek = FakeDeepSeek([reply])
        req = server._ChatRequest(model='deepseek-chat', messages=[{'role': 'user', 'content': 'go'}],
                                  tools=[{'type': 'function', 'function': {'name': name, 'parameters': {}}}
                                         for name in TOOL_NAMES] if tools else None)
        return (await server.chat_completions(req))['choices'][0]

    async def test_bare_call_reaches_the_client(self):
        reply = 'Checking.\n{"name": "shell", "arguments": {"command": "ls"}}'
        choice = await self.complete(reply, tools=True)
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(choice['message']['tool_calls'][0]['function']['arguments'], '{"command": "ls"}')
        choice = await self.complete(reply, tools=False)
        self.assertEqual(choice['message']['content'], reply)

    async def test_apk_transcript_calls_reach_nonstreaming_client(self):
        for reply in APK_CALLS:
            choice = await self.complete(reply, tools=True)
            self.assertEqual(choice['finish_reason'], 'tool_calls')
            self.assertEqual(choice['message']['content'], 'Checking.')
            calls = choice['message']['tool_calls']
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['function']['name'], 'shell')
            self.assertEqual(json.loads(calls[0]['function']['arguments']), {'command': 'python3 --version'})
            choice = await self.complete(reply, tools=False)
            self.assertEqual(choice['message']['content'], reply)

    async def test_malformed_dsml_reaches_nonstreaming_client(self):
        reply = MALFORMED_DSML
        choice = await self.complete(reply, tools=True)
        self.assertEqual(choice['finish_reason'], 'tool_calls')
        self.assertEqual(choice['message']['content'], 'Let me look.')
        self.assertEqual([c['function']['name'] for c in choice['message']['tool_calls']],
                         ['read', 'shell', 'glob'])


class FakeDeepSeek:
    """Answers each turn from a list of replies (a string, an exception, or None for a timeout), recording calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def stream_message(self, message, **kwargs):
        self.calls.append((message, kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if reply is None:
            return
        yield StreamEvent('thinking', 'hmm ')
        yield StreamEvent('text', reply)
        yield StreamEvent('done', response=Response(text=reply, chat_id='chat', search_results=[{'url': 'u'}]))


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_plan(self):
        plan = ('Plan:\n<research_plan>{"title": "RAM", "steps": ['
                '{"action": "search", "query": "DDR5 prices October 2026", "purpose": "current"},'
                '{"action": "fetch", "query": "https://example.com/ram"},'
                '{"action": "fetch", "query": "not a url"},'
                '{"action": "search", "query": ""}, "junk"]}</research_plan>')
        title, steps = parse_plan(plan)
        self.assertEqual(title, 'RAM')
        self.assertEqual([(s['action'], s['query']) for s in steps], [
            ('search', 'DDR5 prices October 2026'), ('fetch', 'https://example.com/ram'), ('search', 'not a url')])
        many = '```json\n{"steps": [' + '{"query": "q"},' * 9 + '{"query": "q"}]}\n```'
        self.assertEqual(len(parse_plan(many)[1]), 6)
        self.assertEqual(parse_plan('No plan here'), ('', []))

    async def test_research_runs_plan_steps_and_report(self):
        plan = ('<research_plan>{"title": "T", "steps": [{"action": "search", "query": "q1"},'
                '{"action": "fetch", "query": "https://example.com"}, {"action": "search", "query": "q3"}]}'
                '</research_plan>')
        ds = FakeDeepSeek([plan, 'found 1', 'found 2', ServerDown('busy'), 'The report'])
        with unittest.mock.patch.object(research, '_fetch_page', return_value='page text'):
            events = [e async for e in research.stream_research(
                ds, 'ram prices', chat_id=None, new_chat=True, system_prompt='sys', files=['a.png'])]

        first, search, fetch, failed, report = [kwargs for _, kwargs in ds.calls]
        self.assertEqual((first['new_chat'], first['system_prompt'], first['files']), (True, 'sys', ['a.png']))
        self.assertTrue(search['search'])
        self.assertEqual(search['chat_id'], 'chat')
        self.assertNotIn('search', fetch)
        self.assertEqual(len(fetch['files']), 1)
        self.assertIn('Steps 3 failed', ds.calls[-1][0])
        self.assertNotIn('files', report)

        text = ''.join(e.content for e in events if e.type == 'text')
        self.assertEqual(text, 'The report')
        streamed = ''.join(e.content for e in events if e.type == 'thinking')
        self.assertIn('found 1', streamed)
        self.assertIn('found 2', streamed)
        done = events[-1].response
        self.assertEqual((done.text, done.chat_id), ('The report', 'chat'))
        self.assertIn('found 2', done.deepthink_content)
        self.assertNotIn('The report', done.deepthink_content)
        self.assertEqual(len(done.search_results), 3)

    async def test_research_falls_back_without_plan_and_stops_on_timeout(self):
        ds = FakeDeepSeek(['I cannot plan', 'found', None])
        events = [e async for e in research.stream_research(ds, 'question')]
        self.assertEqual(ds.calls[1][1]['search'], True)
        self.assertIn('question', ds.calls[1][0])
        self.assertFalse(any(e.type == 'done' for e in events))

    async def test_research_retries_rate_limited_turns(self):
        too_frequent = ServerDown('DeepSeek returned an error: Messages too frequent. Try again later.')
        plan = '<research_plan>{"steps": [{"action": "search", "query": "q1"}]}</research_plan>'
        ds = FakeDeepSeek([too_frequent, plan, too_frequent, 'found', 'Report'])
        with unittest.mock.patch.object(research, 'RATE_LIMIT_DELAYS', (0, 0)):
            events = [e async for e in research.stream_research(ds, 'question', new_chat=True)]
        self.assertEqual(len(ds.calls), 5)
        self.assertTrue(ds.calls[1][1]['new_chat'])
        self.assertEqual(events[-1].response.text, 'Report')
        self.assertIn('found', events[-1].response.deepthink_content)

        ds = FakeDeepSeek([plan, 'found', too_frequent, too_frequent, too_frequent])
        with unittest.mock.patch.object(research, 'RATE_LIMIT_DELAYS', (0, 0)), self.assertRaises(ServerDown):
            [e async for e in research.stream_research(ds, 'question')]

    def test_error_hint_fails_the_response(self):
        state = ResponseState()
        state.feed(None, json.dumps({'v': {'response': {'fragments': [{'type': 'SEARCH', 'content': None}]}}}))
        state.feed('hint', json.dumps({'type': 'error', 'content': 'Messages too frequent. Try again later.'}))
        with self.assertRaisesRegex(ServerDown, 'too frequent'):
            DeepSeek(token='test')._build_response(state, '', 200, 'chat')

    def test_research_model_selection(self):
        def request(**kwargs):
            return server._ChatRequest(messages=[], **kwargs)

        self.assertTrue(server._wants_deepresearch(request(model='deepseek-research')))
        self.assertFalse(server._wants_deepresearch(request(model='deepseek-research', deepresearch=False)))
        self.assertTrue(server._wants_deepresearch(request(model='deepseek-chat', deepresearch=True)))
        self.assertFalse(server._wants_deepresearch(request(model='deepseek-chat')))


if __name__ == '__main__':
    unittest.main()
