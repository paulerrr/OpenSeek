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
    async def test_endpoint_keeps_real_calls_and_chat_then_notes_the_cut(self):
        tools = [{'type': 'function', 'function': {'name': name}} for name in sorted(NAMES)]
        for streaming, known in [(False, True), (True, True), (False, False), (True, False)]:
            closed, sent = [], []

            async def events(message, **kwargs):
                sent.append(kwargs)
                try:
                    if len(sent) == 1:
                        yield StreamEvent('thinking', 'plan')
                        for char in FABRICATED:
                            yield StreamEvent('text', char)
                        yield StreamEvent('done', response=Response(text=FABRICATED, chat_id='bad'))
                    else:
                        yield StreamEvent('text', 'Done.')
                        yield StreamEvent('done', response=Response(text='Done.', chat_id=kwargs['chat_id']))
                finally:
                    closed.append(True)

            ds = Mock()
            ds.stream_message = events
            ds.active_chat_id = 'new'
            expected_chat = 'old' if known else 'new'
            sessions = OrderedDict(s=server._Chat(chat_id='old', instructions_hash=None)) if known else OrderedDict()
            with patch.object(server.state, 'deepseek', ds), patch.object(server.state, 'sessions', sessions), \
                    patch.object(server.state, 'chats', OrderedDict()), patch.object(server, '_save_sessions'):
                request = server._ChatRequest(
                    messages=[{'role': 'user', 'content': 'go'}], stream=streaming, tools=tools)
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
                self.assertEqual(closed, [True])
                # The chat is kept instead of being replayed, and marked as cut
                self.assertEqual(sessions['s'].chat_id, expected_chat)
                self.assertTrue(sessions['s'].cut)

                # The client echoes the reply back with the real result
                call = calls[0]
                assistant = {'role': 'assistant', 'content': content,
                             'tool_calls': [{'id': call['id'], 'type': 'function', 'function': call['function']}]}
                request = server._ChatRequest(messages=[
                    {'role': 'user', 'content': 'go'}, assistant,
                    {'role': 'tool', 'tool_call_id': call['id'], 'content': 'real'}], tools=tools)
                message = (await server.chat_completions(request, 's'))['choices'][0]['message']
                self.assertEqual(message['content'], 'Done.')
                self.assertEqual(sent[1]['chat_id'], expected_chat)
                self.assertFalse(sent[1]['new_chat'])
                self.assertIn(server.CUT_NOTE, sent[1]['system_prompt'])
                self.assertFalse(sessions['s'].cut)

    async def test_cut_without_known_chat_replays_next_turn(self):
        async def events(message, **kwargs):
            for char in FABRICATED:
                yield StreamEvent('text', char)
            yield StreamEvent('done', response=Response(text=FABRICATED, chat_id='bad'))

        ds = Mock()
        ds.stream_message = events
        ds.active_chat_id = None
        sessions = OrderedDict()
        with patch.object(server.state, 'deepseek', ds), patch.object(server.state, 'sessions', sessions), \
                patch.object(server.state, 'chats', OrderedDict()), patch.object(server, '_save_sessions'):
            request = server._ChatRequest(messages=[{'role': 'user', 'content': 'go'}],
                                          tools=[{'type': 'function', 'function': {'name': 'read'}}])
            message = (await server.chat_completions(request, 's'))['choices'][0]['message']
            self.assertEqual([c['function']['name'] for c in message['tool_calls']], ['read'])
            self.assertNotIn('s', sessions)
