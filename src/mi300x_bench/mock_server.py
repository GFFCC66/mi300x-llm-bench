"""Tiny OpenAI-compatible server for tests and CPU dry-runs.

The server streams ``max_tokens`` chunks, then a final usage object. Prompt
tokens are the whitespace word count of the user message, which matches
``synthetic_prompt``. It is not a model and it does not produce benchmark
numbers worth quoting for hardware.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _user_text(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    parts: list[str] = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MockHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (OSError, AttributeError):
            return

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path.rstrip("/") in {"/health", "/v1/models", "/models"}:
            if path.rstrip("/").endswith("models"):
                payload = {
                    "object": "list",
                    "data": [{"id": "mock", "object": "model", "owned_by": "mi300x_bench"}],
                }
            else:
                payload = {"status": "ok"}
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": f"unknown path {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return
        if not isinstance(body, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return
        if self.server.fail_status is not None:
            self._send_json(self.server.fail_status, {"error": "mock forced failure"})
            return

        max_tokens = body.get("max_tokens", 8)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            self._send_json(400, {"error": "max_tokens must be a positive integer"})
            return
        prompt_tokens = max(1, len(_user_text(body).split()))
        stream = body.get("stream", True)
        if stream:
            self._stream(max_tokens, prompt_tokens)
        else:
            text = " ".join(f"t{i}" for i in range(max_tokens))
            self._send_json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_tokens + max_tokens,
                    },
                },
            )

    def _stream(self, max_tokens: int, prompt_tokens: int) -> None:
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if self.server.ttft_s:
            time.sleep(self.server.ttft_s)
        for index in range(max_tokens):
            if index and self.server.itl_s:
                time.sleep(self.server.itl_s)
            # An empty role chunk is sent once, the way vLLM often does, so the
            # client must ignore it when measuring TTFT.
            if index == 0:
                self._write_event(
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            self._write_event(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": f"t{index} "},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        self._write_event(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": max_tokens,
                    "total_tokens": prompt_tokens + max_tokens,
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_event(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        ttft_s: float = 0.0,
        itl_s: float = 0.0,
        fail_status: int | None = None,
    ) -> None:
        self.ttft_s = ttft_s
        self.itl_s = itl_s
        self.fail_status = fail_status
        super().__init__(server_address, _Handler)


class MockServer:
    """Background mock bound to an ephemeral port unless one is given."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        ttft_s: float = 0.0,
        itl_s: float = 0.0,
        fail_status: int | None = None,
    ) -> None:
        self._httpd = MockHTTPServer(
            (host, port),
            ttft_s=ttft_s,
            itl_s=itl_s,
            fail_status=fail_status,
        )
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> MockServer:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def serve_blocking(host: str, port: int, ttft_s: float, itl_s: float) -> None:
    server = MockHTTPServer((host, port), ttft_s=ttft_s, itl_s=itl_s)
    print(f"mock OpenAI server at http://{host}:{port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping mock server", flush=True)
    finally:
        server.server_close()
