"""
In-page request interception, adapted from Better DeepSeek's fetch/XHR patch.

The script is installed on every document before DeepSeek's own code runs. When Python arms it with a
pending mutation, the next chat completion request gets its payload rewritten (hidden prompt prefix,
DeepThink/Search flags), and the raw SSE response is forwarded to Python through a CDP binding as it streams.

It also records the state of files the page uploads (`upload_file` and `fetch_files` responses) in
`window.__openseek.files`, so Python can wait for attachments to finish processing before sending.
"""

BINDING_NAME = "__openseekEmit"

INTERCEPTOR_JS = r"""
(() => {
  if (window.__openseek) return;
  const state = (window.__openseek = { pending: null, files: {}, fileErrors: [] });
  const TARGET = "/api/v0/chat/completion";
  const FILE_PATHS = ["/api/v0/file/upload_file", "/api/v0/file/fetch_files"];

  const emit = (message) => {
    try {
      window.__BINDING__(JSON.stringify(message));
    } catch (e) {}
  };

  const isTarget = (url) => String(url || "").includes(TARGET);
  const isFileRequest = (url) => FILE_PATHS.some((path) => String(url || "").includes(path));

  // Records file states from an upload_file / fetch_files response body.
  const trackFiles = (text) => {
    try {
      const data = JSON.parse(text).data || {};
      if (data.biz_code) {
        state.fileErrors.push(data.biz_msg || `biz_code ${data.biz_code}`);
        return;
      }
      const result = data.biz_data || {};
      for (const file of result.files || [result]) {
        if (file && file.id) state.files[file.id] = file;
      }
    } catch (e) {
      state.fileErrors.push(String(e));
    }
  };

  // Applies the armed mutation (if any) to a request body, returning the new body and the mutation token.
  const mutate = (body) => {
    const pending = state.pending;
    if (!pending || typeof body !== "string") return { body, token: null };
    state.pending = null;
    try {
      const payload = JSON.parse(body);
      if (pending.prefix) payload.prompt = pending.prefix + (payload.prompt || "");
      if (pending.thinking !== null) payload.thinking_enabled = pending.thinking;
      if (pending.search !== null) payload.search_enabled = pending.search;
      emit({
        token: pending.token,
        kind: "request",
        chat_id: payload.chat_session_id || null,
        ref_file_ids: payload.ref_file_ids || [],
      });
      return { body: JSON.stringify(payload), token: pending.token };
    } catch (e) {
      emit({ token: pending.token, kind: "error", error: String(e) });
      return { body, token: pending.token };
    }
  };

  const originalOpen = XMLHttpRequest.prototype.open;
  const originalSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (method, url) {
    this.__openseekUrl = url;
    return originalOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function (body) {
    if (isFileRequest(this.__openseekUrl)) {
      this.addEventListener("load", () => trackFiles(this.responseText));
      return originalSend.apply(this, arguments);
    }
    if (!isTarget(this.__openseekUrl)) return originalSend.apply(this, arguments);

    const { body: nextBody, token } = mutate(body);
    if (token) {
      let offset = 0;
      const flush = () => {
        const text = this.responseText || "";
        if (text.length > offset) {
          emit({ token, kind: "chunk", data: text.slice(offset) });
          offset = text.length;
        }
      };
      this.addEventListener("progress", flush);
      this.addEventListener("loadend", () => {
        flush();
        emit({ token, kind: "done", status: this.status });
      });
    }
    return originalSend.call(this, nextBody);
  };

  const originalFetch = window.fetch;
  window.fetch = async function (input, init) {
    const url = typeof input === "string" || input instanceof URL ? String(input) : input && input.url;
    if (isFileRequest(url)) {
      const response = await originalFetch.apply(this, arguments);
      response.clone().text().then(trackFiles, () => {});
      return response;
    }
    if (!isTarget(url) || !init || typeof init.body !== "string") return originalFetch.apply(this, arguments);

    const { body: nextBody, token } = mutate(init.body);
    const response = await originalFetch.call(this, input, { ...init, body: nextBody });
    if (token && response.body) {
      (async () => {
        const reader = response.clone().body.getReader();
        const decoder = new TextDecoder();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (value) emit({ token, kind: "chunk", data: decoder.decode(value, { stream: !done }) });
            if (done) break;
          }
        } finally {
          emit({ token, kind: "done", status: response.status });
        }
      })();
    }
    return response;
  };
})();
""".replace("__BINDING__", BINDING_NAME)
