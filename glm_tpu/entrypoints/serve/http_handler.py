"""The HTTP request handler of the chat UI and the ``/v1`` API (served on loopback only)."""

from http.server import BaseHTTPRequestHandler
from importlib import resources
import json
import secrets
from urllib.parse import urlsplit

from glm_tpu.entrypoints.openai import protocol


UI_PACKAGE = "glm_tpu.entrypoints.ui"  # the page, script and style sheet: its package data static/*


def handler(store, chat=None, models=None, token=None):
    """The handler class: the browser workspace over ``store`` (a ``ConversationStore``) and ``/v1`` over ``chat``
    (``OpenAIServingChat``) and ``models`` (``OpenAIServingModels``) for bearer ``token`` (``None``: no API)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not put private questions in access logs.

        def respond(self, code, value, kind="application/json"):
            raw = json.dumps(value).encode() if kind == "application/json" else value
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                (
                    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
                ),
            )
            self.end_headers()
            self.wfile.write(raw)

        def bearer(self):
            # The browser UI is same-origin; API clients authenticate with the local key.
            header = self.headers.get("Authorization", "")
            offered = header[7:] if header.startswith("Bearer ") else ""
            return token is not None and secrets.compare_digest(offered, token)

        def stream(self, source):
            try:
                first = next(source)
            except protocol.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except StopIteration:
                return self.respond(500, dict(error=dict(message="empty stream")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(first)
                self.wfile.flush()
                for chunk in source:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except protocol.ApiError as exc:
                # Headers are already sent; report in-band and end the stream.
                self.wfile.write(b"data: " + json.dumps(exc.body()).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client left; the admitted request still finishes.

        def allowed(self):
            # Loopback binding plus Host/Origin checks prevent DNS rebinding and CSRF.
            host = self.headers.get("Host", "")
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            return host in expected and (origin is None or origin == "http://" + host)

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, dict(error="Local origin required."))
            path = urlsplit(self.path).path
            if path.startswith("/v1/"):
                if not self.bearer():
                    return self.respond(
                        401, dict(error=dict(message="A local API key is required.", type="invalid_request_error"))
                    )
                if path != "/v1/models":
                    return self.respond(404, dict(error=dict(message="Not found.", type="invalid_request_error")))
                return self.respond(200, models.models())
            if path == "/api/state":
                return self.respond(200, store.snapshot())
            assets = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
            }
            if path not in assets:
                return self.respond(404, dict(error="Not found."))
            name, kind = assets[path]
            self.respond(200, resources.files(UI_PACKAGE).joinpath("static", name).read_bytes(), kind)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path.startswith("/v1/"):
                return self.completions(path)
            if not self.allowed() or self.headers.get("X-GLM-UI") != "1":
                return self.respond(403, dict(error="Local UI required."))
            if self.path != "/api/chat":
                return self.respond(404, dict(error="Not found."))
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 256000:
                    raise ValueError("Message is too large.")
                self.connection.settimeout(10)
                data = json.loads(self.rfile.read(size))
                if type(data) is not dict:
                    raise ValueError("Expected an object.")
                result = store.change(data)
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=str(exc)))
            self.respond(200, result)

        def completions(self, path):
            if not self.allowed():
                return self.respond(
                    403, dict(error=dict(message="Local origin required.", type="invalid_request_error"))
                )
            if not self.bearer():
                return self.respond(
                    401, dict(error=dict(message="A local API key is required.", type="invalid_request_error"))
                )
            if path != "/v1/chat/completions":
                return self.respond(404, dict(error=dict(message="Not found.", type="invalid_request_error")))
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4 << 20:
                    raise protocol.ApiError("request body is empty or too large")
                self.connection.settimeout(30)
                data = json.loads(self.rfile.read(size))
            except protocol.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=dict(message=str(exc), type="invalid_request_error")))
            self.connection.settimeout(None)
            if type(data) is dict and data.get("stream"):
                return self.stream(chat.stream(data))
            try:
                return self.respond(200, chat.completion(data))
            except protocol.ApiError as exc:
                return self.respond(exc.status, exc.body())
            except (ValueError, OSError) as exc:
                return self.respond(400, dict(error=dict(message=str(exc), type="invalid_request_error")))

    return Handler
