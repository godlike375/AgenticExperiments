import json
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

UPSTREAM_URL = "https://opencode.ai/zen/v1/chat/completions"
API_KEY = "public"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8000


def _random_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex}"


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = [
                "hy3-free", "big-pickle", "deepseek-v4-flash-free",
                "mimo-v2.5-free", "nemotron-3-ultra-free",
                "north-mini-code-free", "glm-4.5v-free",
            ]
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": 0, "owned_by": "opencode"}
                    for m in models
                ],
            })
        else:
            self._send_json(200, {"status": "ok", "proxy": "opencode-zen"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request"}})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"

        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": {"message": f"invalid json: {e}", "type": "invalid_request"}})
            return

        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {API_KEY}",
            "X-Opencode-Client": "desktop",
            "X-Opencode-Project": "proxy",
            "X-Opencode-Session": _random_id("ses"),
            "X-Opencode-Request": _random_id("msg"),
            "User-Agent": "opencode/local ai-sdk/provider-utils/4.0.23 runtime/node.js/24",
        }

        req = Request(UPSTREAM_URL, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(req, timeout=300) as resp:
                upstream = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(upstream)))
            self.end_headers()
            self.wfile.write(upstream)
        except HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
        except URLError as e:
            self._send_json(502, {"error": {"message": f"upstream error: {e.reason}", "type": "upstream_error"}})

    def log_message(self, fmt, *args):
        print("[proxy]", fmt % args)


if __name__ == "__main__":
    print(f"OpenCode Zen proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Upstream: {UPSTREAM_URL}")
    HTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
