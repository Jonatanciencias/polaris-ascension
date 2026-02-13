#!/usr/bin/env python3
"""Local one-shot webhook receiver for Week22 Block2 live-cutover validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class _WebhookHandler(BaseHTTPRequestHandler):
    server_version = "Week22WebhookReceiver/1.0"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        payload: Any
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")}

        envelope = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "payload": payload,
        }
        self.server.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.server.output_path.write_text(json.dumps(envelope, indent=2) + "\n")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}\n')

        self.server.request_count += 1
        if self.server.request_count >= self.server.max_requests:
            self.server.keep_running = False

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-shot local webhook receiver.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--output-json",
        default="research/breakthrough_lab/week22_controlled_rollout/week22_block2_live_webhook_capture.json",
    )
    parser.add_argument("--max-requests", type=int, default=1)
    args = parser.parse_args()

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    server = HTTPServer((args.host, args.port), _WebhookHandler)
    server.output_path = output_path
    server.max_requests = int(args.max_requests)
    server.request_count = 0
    server.keep_running = True

    print(f"Webhook receiver listening on http://{args.host}:{args.port}")
    print(f"Output JSON: {output_path}")

    while server.keep_running:
        server.handle_request()

    print("Webhook receiver closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
