"""Tiny static file server with permissive CORS, used to hand the code zip to
the Colab output frame (the frame fetches it and feeds the upload widget).

    python tools/serve.py <directory> <port>
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # noqa: D102
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args, **kwargs) -> None:  # keep output quiet
        pass


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    handler = functools.partial(Handler, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"serving {directory} on http://127.0.0.1:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
