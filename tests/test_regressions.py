import asyncio
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
from openseek.internal.tools import ToolCallStreamFilter, build_tools_prompt, parse_tool_calls


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


TOOL_NAMES = {'read', 'grep', 'shell', 'glob', 'edit'}

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

# A parameter tag missing its `parameter name="` prefix
MANGLED_PARAMETER = '''Exploring.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/openseek/internal</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/openseek/openseek.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ limit" string="false">150</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>'''

# A parameter opening tag with an extra slash, like the slashed invoke opener
SLASHED_PARAMETER = MANGLED_PARAMETER.replace(
    '<｜｜DSML｜｜ limit" string', '</｜｜DSML｜｜ parameter name="limit" string')

# Parameters closed by a nameless DSML tag, with the invoke left unclosed before the calls closing tag
BARE_PARAMETER_CLOSE = '''Wiring the gate.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="edit">
<｜｜DSML｜｜ parameter name="newString" string="true">        if not gate(re.search(rf"<@!?{self.user.id}>", text)):
            return
</｜DSML｜｜ parameter>
<｜DSML｜｜ parameter name="oldString" string="true">        direct = 1
</｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="path" string="true">/x/bot.py</｜｜DSML｜｜>
</｜DSML｜｜ calls>
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/bot.py</｜｜DSML｜｜>
<｜｜DSML｜｜ parameter name="offset" string="false">340</｜｜DSML｜｜>
</｜DSML｜｜ calls>'''

# JSON calls followed by a spaced closing tag
SPACED_CLOSE = '''Reading.

<tool_call>
{"name": "read", "arguments": {"path": "/x/README.md"}}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/x/server.py"}}
</tool_call>
</tool calls>'''

# A JSON call followed by DSML parameters whose invoke opening tag is missing
HEADLESS_INVOKE = '''Reading.

<tool_call>
{"name": "read", "arguments": {"path": "/x/openseek.py"}}
</tool_call>
<｜｜DSML｜｜ parameter name="path">/x/internal</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</tool_call>
</｜｜DSML｜｜ calls>'''

# Calls wrapped in <call> instead of <tool_call>
SHORT_CALL_TAGS = '''Reading.

<tool_call>
{"name": "read", "arguments": {"path": "/x/interceptor.py"}}
</tool_call>
<call>
{"name": "read", "arguments": {"path": "/x/stream.py"}}
</call>
<call>
{"name": "read", "arguments": {"path": "/x/objects.py"}}
</call>
</｜｜DSML｜｜ calls>'''

# A DSML calls wrapper closed with a second opening tag
REOPENED_WRAPPER = '''Reading.

<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/tools.py</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/research.py</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ calls>'''

# Parameters closed twice, and a last invoke with a slashed opener and no closing tag
DOUBLE_PARAMETER_CLOSE = '''Checking.

<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/.gitignore</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="glob">
<｜｜DSML｜｜ parameter name="pattern" string="true">**/{TODO,TODO.md}</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/tests</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
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


# Calls followed by a closing tool_result tag echoed from the result format
ECHOED_RESULT_CLOSE = '''Reading.

<tool_call>
{"name": "read", "arguments": {"path": "/x/README.md"}}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"path": "/x/server.py"}}
</tool_call>
</tool_result>'''


APK_CALLS = [
    'Checking.\n<tool_call>\n'
    '{"name": "shell", "arguments": {"command": "python3 --version"}}\n'
    + closing + '\n<｜｜DSML｜｜ calls>'
    for closing in ['</call-tool>', '']
]
APK_CALLS.append(APK_CALLS[1].replace('<tool_call>\n', ''))

# Parameters closed without "</", and the invoke left open before a second wrapper opener that ends the reply
SLASHLESS_CLOSE = (
    'Running.\n<｜｜DSML｜｜ calls> <｜｜DSML｜｜ invoke name="shell"> '
    '<｜｜DSML｜｜ parameter name="command" string="true">python t.py</｜｜DSML｜｜ parameter> '
    '<｜｜DSML｜｜ parameter name="workdir" string="true">C:\\New folder｜｜DSML｜｜ parameter> '
    '<｜｜DSML｜｜ parameter name="timeout" string="false">60000｜｜DSML｜｜ parameter> <｜｜DSML｜｜ calls>'
)

# A parameter closed by a closing tag that also names the next parameter
MERGED_PARAMETER_TAGS = (
    'Checking.\n<｜｜DSML｜｜ calls> <｜｜DSML｜｜ invoke name="glob"> '
    '<｜｜DSML｜｜ parameter name="pattern" string="true">*</｜｜DSML｜｜ parameter> </｜｜DSML｜｜ parameter> '
    '</｜｜DSML｜｜ invoke> <｜｜DSML｜｜ invoke name="shell"> '
    '<｜｜DSML｜｜ parameter name="command" string="true">ls\n-la</｜｜DSML｜｜ parameter name="workdir" string="true">'
    'C:\\New folder</｜｜DSML｜｜ parameter> </｜｜DSML｜｜ calls>'
)

# Parameter tags missing " name" before the name attribute, closed by nameless tags
NAMELESS_ATTRIBUTE = '''Re-issuing it.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="edit">
<｜｜DSML｜｜ parameter name="newString" string="true">    code = str(raw.get("code", ""))
    return code</｜DSML｜｜>
<｜｜DSML｜｜ parameter="oldString" string="true">    return raw</｜DSML｜｜>
<｜｜DSML｜｜ parameter="path" string="true">C:\\New folder\\exposure.py</｜｜DSML｜｜>
</｜｜DSML｜｜>
</｜｜DSML｜｜ calls>'''

# An empty opening tag before the first parameter
EMPTY_OPEN_TAG = '''Threading it through.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="edit">
<｜｜DSML｜｜>
<｜｜DSML｜｜ parameter name="newString" string="true">b</｜DSML｜｜>
<｜｜DSML｜｜ parameter name="oldString" string="true">a</｜DSML｜｜>
<｜｜DSML｜｜ parameter name="path" string="true">/x/exposure.py</｜DSML｜｜>
</｜｜DSML｜｜>
</｜｜DSML｜｜ calls>'''

# A calls opener repeated between invokes, with the second invoke closed by a stray parameter tag
REPEATED_OPENER = '''Checking the code.
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/exposure.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="offset" string="false">226</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="read">
<｜｜DSML｜｜ parameter name="path" string="true">/x/exposure.py</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="offset" string="false">458</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ calls>'''


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_malformed_calls(self):
        for text, names, prefix in [(MALFORMED_DSML, ['read', 'shell', 'glob'], 'Let me look.'),
                                    (MIXED_CALL, ['read'], 'Reading.'),
                                    (JSON_THEN_DSML, ['shell', 'read'], 'Adding a style.'),
                                    (MANGLED_PARAMETER, ['read', 'read'], 'Exploring.'),
                                    (SLASHED_PARAMETER, ['read', 'read'], 'Exploring.'),
                                    (BARE_PARAMETER_CLOSE, ['edit', 'read'], 'Wiring the gate.'),
                                    (SPACED_CLOSE, ['read', 'read'], 'Reading.'),
                                    (HEADLESS_INVOKE, ['read'], 'Reading.'),
                                    (SHORT_CALL_TAGS, ['read', 'read', 'read'], 'Reading.'),
                                    (REOPENED_WRAPPER, ['read', 'read'], 'Reading.'),
                                    (DOUBLE_PARAMETER_CLOSE, ['read', 'glob', 'read'], 'Checking.'),
                                    (ECHOED_RESULT_CLOSE, ['read', 'read'], 'Reading.'),
                                    (SLASHLESS_CLOSE, ['shell'], 'Running.'),
                                    (MERGED_PARAMETER_TAGS, ['glob', 'shell'], 'Checking.'),
                                    (NAMELESS_ATTRIBUTE, ['edit'], 'Re-issuing it.'),
                                    (EMPTY_OPEN_TAG, ['edit'], 'Threading it through.'),
                                    (REPEATED_OPENER, ['read', 'read'], 'Checking the code.'),
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
        calls = parse_tool_calls(BARE_PARAMETER_CLOSE, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[1]['function']['arguments']), {'path': '/x/bot.py', 'offset': 340})
        self.assertTrue(json.loads(calls[0]['function']['arguments'])['newString'].endswith('return\n'))
        for text in [MANGLED_PARAMETER, SLASHED_PARAMETER]:
            calls = parse_tool_calls(text, TOOL_NAMES)[1]
            self.assertEqual(json.loads(calls[1]['function']['arguments']),
                             {'path': '/x/openseek/openseek.py', 'limit': 150})
        calls = parse_tool_calls(SLASHLESS_CLOSE, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[0]['function']['arguments']),
                         {'command': 'python t.py', 'workdir': 'C:\\New folder', 'timeout': 60000})
        calls = parse_tool_calls(MERGED_PARAMETER_TAGS, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[1]['function']['arguments']),
                         {'command': 'ls\n-la', 'workdir': 'C:\\New folder'})
        calls = parse_tool_calls(NAMELESS_ATTRIBUTE, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[0]['function']['arguments']),
                         {'newString': '    code = str(raw.get("code", ""))\n    return code',
                          'oldString': '    return raw', 'path': 'C:\\New folder\\exposure.py'})
        calls = parse_tool_calls(EMPTY_OPEN_TAG, TOOL_NAMES)[1]
        self.assertEqual(json.loads(calls[0]['function']['arguments']),
                         {'newString': 'b', 'oldString': 'a', 'path': '/x/exposure.py'})
        calls = parse_tool_calls(REPEATED_OPENER, TOOL_NAMES)[1]
        self.assertEqual([json.loads(c['function']['arguments'])['offset'] for c in calls], [226, 458])

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
                     'Example </tool calls>',
                     'Example </tool_result>',
                     '<｜｜DSML｜｜ parameter name="path">/x</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>',
                     '<call>{"name":"unknown","arguments":{}}</call>',
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

    async def test_unescaped_quotes_in_single_string_argument_are_repaired(self):
        # From an opencode transcript: inner double quotes and \| left unescaped in a shell command
        command = 'cd /x && grep -n "scheduleSave\\|beforeunload" app.js && echo "--- \\"y\\" ---"'
        body = '{"name": "shell", "arguments": {"command": "%s"}}' % command.replace('\\"y\\"', '\\\\"y\\\\"')
        for closing in ['</tool_call>', '</｜｜DSML｜｜ calls>']:
            text = 'Let me look.\n<tool_call>\n' + body + '\n' + closing
            content, remembered, _ = await self.collect(text)
            self.assertEqual(content.strip(), 'Let me look.')
            self.assertEqual(json.loads(remembered[0][2][0]['function']['arguments']), {'command': command})

    async def test_unrepairable_calls_stay_text(self):
        for text in ['<tool_call>{"name": "edit", "arguments": {"path": "a", "oldString": "x"y"}}</tool_call>',
                     '<tool_call>{"name": "shell", "arguments": {"command": "a"b"}} trailing</tool_call>',
                     '<tool_call>{"name": "unknown", "arguments": {"command": "a"b"}}</tool_call>']:
            content, remembered, _ = await self.collect(text)
            self.assertEqual(content, text)
            self.assertEqual(remembered[0][2], [])

    async def test_dsml_string_arguments_closed_by_calls_tag(self):
        # From an opencode transcript: no </parameter> or </invoke>, and an extra calls closing tag
        edit = {'path': '/x/app.js', 'oldString': 'a\n', 'newString': '"b"'}
        text = ('Let me make the edits.\n<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="edit">\n'
                '<｜｜DSML｜｜ parameter name="arguments" string="true">' + json.dumps(edit)
                + '</｜｜DSML｜｜ calls>\n</｜｜DSML｜｜ calls>')
        for variant in [text, text.replace('</｜｜DSML｜｜ calls>\n', '</｜｜DSML｜｜ parameter>\n</invoke>\n', 1)]:
            content, remembered, _ = await self.collect(variant)
            self.assertEqual(content.strip(), 'Let me make the edits.')
            self.assertEqual(json.loads(remembered[0][2][0]['function']['arguments']), edit)
            self.assertEqual(parse_tool_calls(variant, ['edit'])[0], 'Let me make the edits.')

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

    def test_deepthink_answer_appends_after_path_without_op(self):
        # Captured from a DeepThink reply: the answer's path arrives without `o` and must keep appending
        lines = [
            {"v": {"response": {"fragments": [{"type": "THINK", "content": "We"}]}}},
            {"p": "response/fragments/-1/content", "o": "APPEND", "v": " need"},
            {"v": " answer."},
            {"p": "response/fragments/-1/elapsed_secs", "o": "SET", "v": 1.15},
            {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "hi"}]},
            {"p": "response/fragments/-1/content", "v": " m"},
            {"v": "ike."},
            {"p": "response", "o": "BATCH", "v": [{"p": "quasi_status", "v": "FINISHED"}]},
        ]
        state = ResponseState()
        deltas = [delta for line in lines for delta in state.feed(None, json.dumps(line))]
        self.assertEqual(state.text_of('THINK'), 'We need answer.')
        self.assertEqual(state.text_of('RESPONSE'), 'hi mike.')
        self.assertEqual(''.join(text for kind, text in deltas if kind == 'RESPONSE'), 'hi mike.')
        self.assertEqual(state.fragments[0]['elapsed_secs'], 1.15)

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

    async def test_tool_permissions_on_reused_chat(self):
        fake = FakeDeepSeek(['OK', 'OK', 'OK'])
        server.state.deepseek = fake
        messages = [{'role': 'user', 'content': 'Read the project.'}]
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {}}}]
        for choice in ['auto', 'required', 'none']:
            result = await server.chat_completions(server._ChatRequest(
                messages=messages, tools=tools, tool_choice=choice))
            messages.extend([result['choices'][0]['message'], {'role': 'user', 'content': 'Continue.'}])
        self.assertFalse(fake.calls[1][1]['new_chat'])
        self.assertIn('must call at least one tool', fake.calls[1][1]['system_prompt'])
        self.assertNotIn('Available tools (JSON)', fake.calls[1][1]['system_prompt'])
        self.assertIn('No tools are available', fake.calls[2][1]['system_prompt'])

    def test_tool_schema_order_does_not_change_instructions(self):
        first = {'name': 'read', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}
        second = {'name': 'edit', 'parameters': {}}
        prompt = build_tools_prompt([first, second])
        self.assertEqual(prompt, build_tools_prompt([second, dict(reversed(list(first.items())))]))
        encoded = prompt.split('Available tools (JSON):\n')[1]
        self.assertEqual(json.loads(encoded)[1]['parameters'], first['parameters'])
        self.assertLess(len(encoded), len(json.dumps([first, second], indent=2)))

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


def dsml_call(name, parameters):
    """A DSML call; `parameters` is a list of (name, string attribute or None, value)."""
    body = ''.join(
        f'<｜｜DSML｜｜ parameter name="{key}"' + (f' string="{string}"' if string else '') + f'>{value}</｜｜DSML｜｜ parameter>\n'
        for key, string, value in parameters
    )
    return f'<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="{name}">\n{body}</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>'


class DsmlSchemaTests(unittest.TestCase):
    # Edge cases from vLLM's DeepSeek V4 DSML parser (vllm-project/vllm#41240)
    TOOLS = {
        'shell': {'type': 'object', 'properties': {'command': {'type': 'string'}, 'timeout': {'type': 'integer'}}},
        'run': {'type': 'object', 'properties': {'arguments': {'type': 'object'}}},
        'transform': {'type': 'object', 'properties': {'input': {'type': 'object'}}},
        'bare': None,
    }

    def arguments(self, text, tools=None):
        calls = parse_tool_calls(text, self.TOOLS if tools is None else tools)[1]
        self.assertEqual(len(calls), 1)
        return json.loads(calls[0]['function']['arguments'])

    def test_wrappers_unwrap_when_not_schema_fields(self):
        for wrapper in ['arguments', 'input']:
            for string in ['true', 'false']:
                text = dsml_call('shell', [(wrapper, string, '{"command": "ls", "timeout": 5}')])
                self.assertEqual(self.arguments(text), {'command': 'ls', 'timeout': 5})

    def test_real_wrapper_named_fields_are_kept(self):
        self.assertEqual(self.arguments(dsml_call('run', [('arguments', 'false', '{"x": 1}')])), {'arguments': {'x': 1}})
        self.assertEqual(self.arguments(dsml_call('transform', [('input', 'false', '{"x": 1}')])), {'input': {'x': 1}})

    def test_wrapper_with_unknown_keys_is_kept(self):
        text = dsml_call('shell', [('arguments', 'false', '{"cmd": "ls"}')])
        self.assertEqual(self.arguments(text), {'arguments': {'cmd': 'ls'}})

    def test_without_schema_only_arguments_unwraps(self):
        for tools in [self.TOOLS, {'bare'}]:
            self.assertEqual(self.arguments(dsml_call('bare', [('arguments', 'false', '{"x": 1}')]), tools), {'x': 1})
            self.assertEqual(self.arguments(dsml_call('bare', [('input', 'false', '{"x": 1}')]), tools),
                             {'input': {'x': 1}})

    def test_string_schema_keeps_text_values(self):
        text = dsml_call('shell', [('command', None, '123'), ('timeout', None, '30')])
        self.assertEqual(self.arguments(text), {'command': '123', 'timeout': 30})
        self.assertEqual(self.arguments(text, {'shell'}), {'command': 123, 'timeout': 30})
        # string="true" stays literal whatever the schema says
        self.assertEqual(self.arguments(dsml_call('shell', [('timeout', 'true', '30')])), {'timeout': '30'})

    def test_streamed_response_uses_schema(self):
        async def events():
            text = dsml_call('shell', [('input', 'false', '{"command": "true"}')])
            for char in text:
                yield StreamEvent('text', char)
            yield StreamEvent('done', response=Response(text=text, chat_id='chat'))

        async def collect():
            remembered = []
            async for _ in server._stream_completion(events(), 'id', 0, 'deepseek-chat', self.TOOLS,
                                                     lambda *args: remembered.append(args)):
                pass
            return remembered

        remembered = asyncio.run(collect())
        self.assertEqual(json.loads(remembered[0][2][0]['function']['arguments']), {'command': 'true'})



class DsmlToolCallsWrapperTests(unittest.TestCase):
    # From deepseek-ai/DeepSeek-V4-Pro discussion 209: V4 names the wrapper tool_calls, sometimes with ASCII bars
    def test_tool_calls_wrapper_is_parsed_and_removed(self):
        for bar in ['||', '｜']:
            dsml = f'<{bar}DSML{bar}'
            parameter = f'{dsml}parameter name="query" string="true">bond detail fields display</{bar}DSML{bar}parameter>'
            invoke = f'{dsml}invoke name="search">{parameter}'
            for text in [f'Let me search.{dsml}tool_calls>{invoke}</{bar}DSML{bar}invoke></{bar}DSML{bar}tool_calls>',
                         f'Let me search.\n{dsml}tool_calls>\n{invoke}\n</{bar}DSML{bar}tool_calls>']:
                with self.subTest(text=text):
                    content, calls = parse_tool_calls(text, {'search'})
                    self.assertEqual(content, 'Let me search.')
                    self.assertEqual([call['function']['name'] for call in calls], ['search'])
                    self.assertEqual(json.loads(calls[0]['function']['arguments']), {'query': 'bond detail fields display'})
                    stream = ToolCallStreamFilter({'search'})
                    # Streaming passes the newline before the block through, as it does for other wrappers
                    streamed = ''.join(stream.feed(char) for char in text) + stream.finish()
                    self.assertEqual(streamed.strip(), 'Let me search.')

if __name__ == '__main__':
    unittest.main()
