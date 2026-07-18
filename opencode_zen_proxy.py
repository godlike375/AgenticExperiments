import json
import uuid
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))

UPSTREAM_URL = "https://opencode.ai/zen/v1/chat/completions"
API_KEY = "public"

ZEN_MODELS = [
    "hy3-free", "big-pickle", "deepseek-v4-flash-free",
    "mimo-v2.5-free", "nemotron-3-ultra-free",
    "north-mini-code-free",
]


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
    # Native Ollama /api/chat uses {model, messages:[{role,content}], stream, options}
    messages = body.get("messages", [])
    out = {
        "model": body.get("model", "hy3-free"),
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
    # Convert a streaming OpenAI chunk back to Ollama /api/chat style.
    model = body.get("model", "hy3-free")
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
    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_raw(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _forward(self, body):
        data = json.dumps(body).encode("utf-8")
        req = Request(UPSTREAM_URL, data=data, headers=_zen_headers(), method="POST")
        try:
            with urlopen(req, timeout=300) as resp:
                upstream = resp.read()
            return resp.status, upstream
        except HTTPError as e:
            return e.code, e.read()
        except URLError as e:
            return 502, json.dumps({
                "error": {"message": f"upstream error: {e.reason}", "type": "upstream_error"}
            }).encode("utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "opencode"}
                    for m in ZEN_MODELS
                ],
            })
        elif path in ("/api/tags", "/api/models"):
            # Native Ollama model listing
            self._send_json(200, {
                "models": [{"name": m, "model": m} for m in ZEN_MODELS]
            })
        elif path in ("/api/v1/models",):
            # Native LM Studio model listing (OpenAI-style)
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "lmstudio"}
                    for m in ZEN_MODELS
                ],
            })
        else:
            self._send_json(200, {"status": "ok", "proxy": "opencode-zen"})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")

        if path in ("/v1/chat/completions", "/chat/completions"):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request"}})
                return
            status, upstream = self._forward(body)
            self._send_raw(status, upstream)
            return

        if path in ("/api/chat", "/api/generate"):
            # Native Ollama endpoints -> normalize to OpenAI then back
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request"}})
                return
            oai = _ollama_to_openai(body)
            status, upstream = self._forward(oai)
            if body.get("stream"):
                # Streaming: pass through SSE but convert each OpenAI chunk
                self.send_response(status)
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
                        "model": oai.get("model", "hy3-free"),
                        "message": {"role": "assistant", "content": msg.get("content", "")},
                        "done": True,
                    })
                except Exception:
                    self._send_raw(status, upstream)
            return

        if path in ("/api/v1/chat", "/api/v1/chat/completions"):
            # Native LM Studio chat endpoint (OpenAI-compatible payload)
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request"}})
                return
            status, upstream = self._forward(body)
            self._send_raw(status, upstream)
            return

        if path in ("/api/v1/models/load",):
            # LM Studio model loading stub - proxy serves remote zen models
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            except Exception:
                body = {}
            model = body.get("model_key") or body.get("model") or ZEN_MODELS[0]
            self._send_json(200, {
                "success": True,
                "is_loaded": True,
                "loading_context": {"model_key": model, "note": "served via opencode-zen proxy"},
            })
            return

        if path in ("/api/v1/models/download",):
            # LM Studio download stub
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            except Exception:
                body = {}
            self._send_json(200, {
                "success": True,
                "job_id": _random_id("dl"),
                "note": "served via opencode-zen proxy",
            })
            return

        if path.startswith("/api/v1/models/download/status/"):
            # LM Studio download status stub
            job_id = path.rsplit("/", 1)[-1]
            self._send_json(200, {
                "success": True,
                "job_id": job_id,
                "status": "completed",
                "progress": 1.0,
                "note": "served via opencode-zen proxy",
            })
            return

        self._send_json(404, {"error": {"message": "not found", "type": "invalid_request"}})

    def log_message(self, fmt, *args):
        print("[proxy]", fmt % args)


if __name__ == "__main__":
    print(f"OpenCode Zen proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Upstream: {UPSTREAM_URL}")
    print("Accepts: OpenAI-compatible (LM Studio / Ollama / llama.cpp),")
    print("         native Ollama /api/chat, native LM Studio /api/v1/*")
    HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
