const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync('openseek/internal/interceptor.py', 'utf8')
  .split('INTERCEPTOR_JS = r"""')[1].split('""".replace')[0]
  .replaceAll('__BINDING__', '__openseekEmit');

function setup(fetch) {
  const events = [];
  class XHR {
    listeners = {};
    status = 0;
    responseText = '';
    open() {}
    send() {}
    addEventListener(kind, callback) { this.listeners[kind] = callback; }
  }
  const window = { fetch, __openseekEmit: (text) => events.push(JSON.parse(text)) };
  vm.runInNewContext(source, { window, XMLHttpRequest: XHR, URL, TextDecoder });
  window.__openseek.pending = { token: 'test', thinking: false, search: false };
  return { window, events, XHR };
}

test('fetch rejection emits a transport failure', async () => {
  const { window, events } = setup(async () => { throw Error('offline'); });
  await assert.rejects(window.fetch('/api/v0/chat/completion', { body: '{}' }), /offline/);
  assert.equal(events.at(-1).kind, 'transport_error');
  assert.equal(events.some(e => e.kind === 'done'), false);
});

test('stream read failure emits failure without done and releases the reader', async () => {
  let released = false;
  let reads = 0;
  const reader = {
    async read() {
      if (reads++ === 0) return { value: new TextEncoder().encode('partial'), done: false };
      throw Error('reset');
    },
    releaseLock() { released = true; },
  };
  const { window, events } = setup(async () => ({
    status: 200, body: {}, clone: () => ({ body: { getReader: () => reader } }),
  }));
  await window.fetch('/api/v0/chat/completion', { body: '{}' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(events.some(e => e.kind === 'chunk'), true);
  assert.equal(events.at(-1).kind, 'transport_error');
  assert.equal(events.some(e => e.kind === 'done'), false);
  assert.equal(released, true);
});

test('XHR abort emits failure before loadend', () => {
  const { XHR, events } = setup(async () => {});
  const xhr = new XHR();
  xhr.open('POST', '/api/v0/chat/completion');
  xhr.send('{}');
  xhr.listeners.abort();
  xhr.listeners.loadend();
  assert.equal(events.at(-2).kind, 'transport_error');
  assert.equal(events.at(-1).status, 0);
});
