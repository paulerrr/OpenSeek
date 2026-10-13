import json
import unittest
from collections import OrderedDict
from unittest.mock import Mock, patch

import server
from openseek.internal.objects import Response, StreamEvent
from openseek.internal.tools import FabricatedResultGuard, ToolCallStreamFilter, parse_tool_calls

NAMES = {'read', 'edit'}
CALL = '<tool_call>\n{"name": "read", "arguments": {"path": "index.html"}}\n</tool_call>\n'
# Shaped like a real transcript: a call, then an invented result the model goes on to reason from
FABRICATED = ('Let me read the file.\n' + CALL + '\n\n</system_instructions>\n\n'
              '<tool_result name="read" id="call_a7d07bfa75aa4c58a959ff75">\n1: <!DOCTYPE html>\n</result>\n'
              'The file is as expected. Now the edit.\n'
              '<tool_call>\n{"name": "edit", "arguments": {"path": "index.html"}}\n</tool_call>\n')


def run(guard, text, size):
    output = ''.join(guard.feed(text[i:i + size]) for i in range(0, len(text), size))
    return output + guard.finish()


class GuardTests(unittest.TestCase):
    def test_cuts_at_fabricated_result_after_call(self):
        variants = [FABRICATED, FABRICATED.replace('</system_instructions>\n\n', ''),
                    FABRICATED.replace('<tool_result', '<result').replace('</system_instructions>\n\n', '')]
        for text in variants:
            for size in [1, 2, 7, 53, len(text)]:
                guard = FabricatedResultGuard(NAMES)
                output = run(guard, text, size)
                self.assertTrue(guard.cut)
                self.assertEqual(output, guard.text)
                self.assertNotIn('result', output)
                content, calls = parse_tool_calls(output, NAMES)
                self.assertEqual(content, 'Let me read the file.')
                self.assertEqual([c['function']['name'] for c in calls], ['read'])

    def test_text_without_preceding_call_is_preserved(self):
        for text in ['Results come back as\n<tool_result name="read">...</tool_result>\n',
                     '</system_instructions>\nhello',
                     CALL + '```xml\n<tool_result name="read">x</tool_result>\n```\n',
                     CALL + 'Results look like <tool_result name="read"> blocks.\n',
                     'Done.\n<results>\n']:
            for size in [1, 7, len(text)]:
                guard = FabricatedResultGuard(NAMES)
                self.assertEqual(run(guard, text, size), text)
                self.assertFalse(guard.cut)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_keeps_real_calls_and_forgets_session(self):
        for streaming in [False, True]:
            closed = []

            async def events(*args, **kwargs):
                try:
                    yield StreamEvent('thinking', 'plan')
                    for char in FABRICATED:
                        yield StreamEvent('text', char)
                    yield StreamEvent('done', response=Response(text=FABRICATED, chat_id='bad'))
                finally:
                    closed.append(True)

            ds = Mock()
            ds.stream_message = events
            request = server._ChatRequest(
                messages=[{'role': 'user', 'content': 'go'}], stream=streaming,
                tools=[{'type': 'function', 'function': {'name': name}} for name in sorted(NAMES)])
            sessions = OrderedDict(s=server._Chat(chat_id='old', instructions_hash=None))
            with patch.object(server.state, 'deepseek', ds), patch.object(server.state, 'sessions', sessions), \
                    patch.object(server.state, 'chats', OrderedDict()), patch.object(server, '_save_sessions'):
                if streaming:
                    response = await server.chat_completions(request, 's')
                    chunks = [part async for part in response.body_iterator]
                    payloads = [json.loads(p[6:]) for p in chunks if p.startswith('data: {')]
                    deltas = [p['choices'][0]['delta'] for p in payloads]
                    content = ''.join(d.get('content', '') for d in deltas)
                    calls = [c for d in deltas for c in d.get('tool_calls', [])]
                    self.assertEqual(payloads[-1]['choices'][0]['finish_reason'], 'tool_calls')
                else:
                    message = (await server.chat_completions(request, 's'))['choices'][0]['message']
                    content, calls = message['content'] or '', message['tool_calls']
                    self.assertEqual(message['reasoning_content'], 'plan')
                self.assertEqual(content.strip(), 'Let me read the file.')
                self.assertEqual([c['function']['name'] for c in calls], ['read'])
                self.assertNotIn('s', sessions)
                self.assertFalse(server.state.chats)
                self.assertEqual(closed, [True])
