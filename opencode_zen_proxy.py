import json
import uuid
import os
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zen_proxy_log.txt")

def _log(msg):
    line = f"{msg}\n"
    print(line, end="")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "1234"))

UPSTREAM_URL = "https://opencode.ai/zen/v1/chat/completions"
API_KEY = "public"

ZEN_MODELS = [
    "hy3-free", "big-pickle", "deepseek-v4-flash-free",
    "mimo-v2.5-free", "nemotron-3-ultra-free",
    "north-mini-code-free",
]

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", ZEN_MODELS[3])


def _sanitize_chunk(chunk):
    if not isinstance(chunk, dict):
        return chunk
    choices = chunk.get("choices", [])
    for choice in choices:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            clean = {}
            for key in ("content", "tool_calls"):
                if key in delta and delta[key] is not None and delta[key] != "":
                    clean[key] = delta[key]
            if clean and "role" in delta and delta["role"] is not None:
                clean["role"] = delta["role"]
            choice["delta"] = clean
        choice.pop("native_finish_reason", None)
        choice.pop("logprobs", None)
    chunk.pop("provider", None)
    chunk.pop("service_tier", None)
    chunk.pop("native_finish_reason", None)
    chunk.pop("usage", None)
    if not choices and "cost" in chunk:
        return None
    has_payload = False
    for c in choices:
        d = c.get("delta", {})
        if isinstance(d, dict):
            if "content" in d or "tool_calls" in d:
                has_payload = True
                break
        if c.get("finish_reason") is not None:
            has_payload = True
            break
    if not has_payload:
        return None
    return chunk


def _random_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex}"


def _zen_headers():
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {API_KEY}",
        "X-Opencode-Client": "desktop",
        "X-Opencode-Project": "proxy",
        "X-Opencode-Session": _random_id("ses"),
        "X-Opencode-Request": _random_id("msg"),
        "User-Agent": "opencode/local ai-sdk/provider-utils/4.0.23 runtime/node.js/24",
    }


def _ollama_to_openai(body):
    messages = body.get("messages", [])
    out = {
        "model": body.get("model", DEFAULT_MODEL),
        "messages": messages,
    }
    if body.get("stream"):
        out["stream"] = True
    if "options" in body and isinstance(body["options"], dict):
        opts = body["options"]
        if "temperature" in opts:
            out["temperature"] = opts["temperature"]
        if "top_p" in opts:
            out["top_p"] = opts["top_p"]
        if "max_tokens" in opts:
            out["max_tokens"] = opts["max_tokens"]
    return out


def _openai_to_ollama(body):
    model = body.get("model", DEFAULT_MODEL)
    choices = body.get("choices", [])
    delta = choices[0].get("delta", {}) if choices else {}
    content = delta.get("content", "")
    done = choices[0].get("finish_reason") is not None if choices else True
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": done,
    }


class Handler(BaseHTTPRequestHandler):
    def _clean_path(self):
        return self.path.split("?", 1)[0].rstrip("/")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status, payload):
        self._send_raw(status, json.dumps(payload).encode("utf-8"))

    def _send_raw(self, status, data, ctype="application/json"):
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sanitize_body(self, body):
        clean = dict(body)
        mt = clean.get("max_tokens")
        if mt is not None and (not isinstance(mt, int) or mt <= 0):
            del clean["max_tokens"]
        st = clean.get("stop")
        if st is not None and isinstance(st, list) and len(st) == 0:
            del clean["stop"]
        if clean.get("tools") and clean.get("model") == "deepseek-v4-flash-free":
            del clean["tools"]
        return clean

    def _upstream_request(self, body):
        body = self._sanitize_body(body)
        data = json.dumps(body).encode("utf-8")
        req = Request(UPSTREAM_URL, data=data, headers=_zen_headers(), method="POST")
        try:
            resp = urlopen(req, timeout=300)
        except HTTPError as e:
            err = e.read()
            _log(f"[proxy]   UPSTREAM ERROR: {e.code} {err.decode('utf-8', errors='replace')[:500]}")
            return e.code, err, e.headers.get("Content-Type", "application/json"), True
        except URLError as e:
            return 502, json.dumps({
                "error": {"message": f"upstream error: {e.reason}", "type": "upstream_error"}
            }).encode("utf-8"), "application/json", True
        return resp.status, resp, resp.headers.get("Content-Type", "text/event-stream"), False

    def _forward(self, body):
        status, upstream, ctype, is_error = self._upstream_request(body)
        if is_error:
            return status, upstream, ctype
        with upstream:
            return status, upstream.read(), ctype

    def _forward_sse(self, body):
        status, upstream, ctype, is_error = self._upstream_request(body)
        if is_error:
            self._send_raw(status, upstream, ctype)
            return
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        sent_done = False
        sent_stop = False
        try:
            for line in upstream:
                text = line.decode("utf-8", errors="replace")
                if text.startswith("data:"):
                    payload = text[len("data:"):].strip()
                    if payload == "[DONE]":
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        sent_done = True
                        continue
                    try:
                        raw_chunk = json.loads(payload)
                        _log(f"[proxy]   RAW: {json.dumps(raw_chunk)[:800]}")
                        is_eos = not raw_chunk.get("choices") and "cost" in raw_chunk
                        clean = _sanitize_chunk(raw_chunk)
                        if clean is None:
                            _log(f"[proxy]   CLEAN: null ({'end-of-stream' if is_eos else 'reasoning-only'})")
                            if is_eos:
                                break
                            continue
                        choices = clean.get("choices", [])
                        if choices and choices[0].get("finish_reason") is not None:
                            if sent_stop:
                                _log("[proxy]   SKIP: duplicate stop chunk")
                                continue
                            sent_stop = True
                        _log(f"[proxy]   CLEAN: {json.dumps(clean)[:800]}")
                        self.wfile.write(("data: " + json.dumps(clean) + "\n\n").encode("utf-8"))
                    except Exception:
                        self.wfile.write(line)
                    self.wfile.flush()
                else:
                    self.wfile.write(line)
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            if not sent_done:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            upstream.close()
            self.close_connection = True

    def _chat_completion(self, body):
        if body.get("stream"):
            self._forward_sse(body)
        else:
            status, upstream, ctype = self._forward(body)
            self._send_raw(status, upstream, ctype)

    def do_OPTIONS(self):
        path = self._clean_path()
        self._log_request("OPTIONS", path)
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _log_request(self, method, path, body=None):
        _log(f"[proxy] {method} {path}")
        if body:
            _log(f"[proxy]   body: {body[:2000]}")
        for h in ("Content-Type", "Authorization", "User-Agent"):
            val = self.headers.get(h)
            if val:
                _log(f"[proxy]   {h}: {val[:200] if h == 'Authorization' else val}")

    def _handle_models_get(self, path):
        if path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "opencode"}
                    for m in ZEN_MODELS
                ],
            })
        elif path in ("/api/tags", "/api/models"):
            self._send_json(200, {
                "models": [{"name": m, "model": m} for m in ZEN_MODELS]
            })
        elif path in ("/api/v1/models", "/api/v0/models"):
            self._send_json(200, {
                "data": [
                    {
                        "id": m, "object": "model", "type": "llm",
                        "publisher": "opencode", "arch": "unknown",
                        "compatibility_type": "gguf", "quantization": "remote",
                        "state": "not-loaded", "max_context_length": 200000,
                        "capabilities": ["tool_use"],
                    }
                    for m in ZEN_MODELS
                ]
            })
        else:
            return False
        return True

    def do_GET(self):
        path = self._clean_path()
        self._log_request("GET", path)
        if not self._handle_models_get(path):
            self._send_json(200, {"status": "ok", "proxy": "opencode-zen"})

    def do_POST(self):
        path = self._clean_path()
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        self._log_request("POST", path, raw.decode("utf-8", errors="replace"))

        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request"}})
            return
        
        if not body.get("model"):
            body["model"] = DEFAULT_MODEL

        if path in ("/v1/chat/completions", "/chat/completions",
                    "/api/v1/chat", "/api/v1/chat/completions",
                    "/api/v0/chat", "/api/v0/chat/completions"):
            self._chat_completion(body)
            return

        if path in ("/api/chat", "/api/generate"):
            oai = _ollama_to_openai(body)
            status, upstream, ctype = self._forward(oai)
            if body.get("stream"):
                self.send_response(status)
                self._cors()
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for line in upstream.decode("utf-8").splitlines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        self.wfile.write(json.dumps(_openai_to_ollama({"model": oai["model"], "choices": [{"delta": {}, "finish_reason": "stop"}]})).encode("utf-8"))
                        self.wfile.write(b"\n")
                        continue
                    try:
                        chunk = json.loads(payload)
                    except Exception:
                        continue
                    self.wfile.write((json.dumps(_openai_to_ollama(chunk)) + "\n").encode("utf-8"))
            else:
                try:
                    oai_resp = json.loads(upstream.decode("utf-8"))
                    msg = oai_resp.get("choices", [{}])[0].get("message", {})
                    self._send_json(status, {
                        "model": oai.get("model", DEFAULT_MODEL),
                        "message": {"role": "assistant", "content": msg.get("content", "")},
                        "done": True,
                    })
                except Exception:
                    self._send_raw(status, upstream)
            return

        if path in ("/api/v1/models/load", "/api/v0/models/load"):
            model = body.get("model_key") or body.get("model") or ZEN_MODELS[0]
            self._send_json(200, {
                "success": True, "is_loaded": True,
                "loading_context": {"model_key": model, "note": "served via opencode-zen proxy"},
            })
            return

        if path in ("/api/v1/models/download", "/api/v0/models/download"):
            self._send_json(200, {"success": True, "job_id": _random_id("dl"), "note": "served via opencode-zen proxy"})
            return

        if path.startswith("/api/v1/models/download/status/") or path.startswith("/api/v0/models/download/status/"):
            self._send_json(200, {
                "success": True, "job_id": path.rsplit("/", 1)[-1],
                "status": "completed", "progress": 1.0, "note": "served via opencode-zen proxy",
            })
            return

        self._send_json(404, {"error": {"message": "not found", "type": "invalid_request"}})

    def log_message(self, fmt, *args):
        _log("[proxy]" + fmt % args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCode Zen proxy")
    parser.add_argument("--default-model", default=DEFAULT_MODEL,
                        help=f"Default model name when client sends empty/missing model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    DEFAULT_MODEL = args.default_model

    _log(f"OpenCode Zen proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    _log(f"Upstream: {UPSTREAM_URL}")
    _log(f"Default model: {DEFAULT_MODEL}")
    _log("Accepts: OpenAI-compatible (LM Studio / Ollama / llama.cpp),")
    _log("         native Ollama /api/chat, native LM Studio /api/v1/*")
    HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
