#!/usr/bin/env python3
"""Local webhook receiver for Week24 Block2 observability validation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


class _WebhookHandler(BaseHTTPRequestHandler):
    server_version = "Week24WebhookReceiver/1.0"

    def _write_capture(self, entry: dict[str, Any]) -> None:
        self.server.capture["requests"].append(entry)
        self.server.capture["summary"]["requests_total"] = len(self.server.capture["requests"])
        self.server.capture["summary"]["posts_total"] = int(self.server.post_count)
        self.server.capture["summary"]["health_checks_total"] = int(self.server.health_count)
        self.server.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.server.output_path.write_text(json.dumps(self.server.capture, indent=2) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        self.server.health_count += 1
        status = 200 if self.path.startswith("/health") else 404
        if self.server.response_delay_ms > 0:
            time.sleep(self.server.response_delay_ms / 1000.0)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": "GET",
            "path": self.path,
            "status": status,
            "headers": {k: v for k, v in self.headers.items()},
            "payload": None,
        }
        self._write_capture(entry)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = {"status": "ok" if status == 200 else "not_found"}
        self.wfile.write((json.dumps(body) + "\n").encode("utf-8"))

        self.server.request_count += 1
        if self.server.request_count >= self.server.max_requests:
            self.server.keep_running = False

    def do_POST(self) -> None:  # noqa: N802
        self.server.post_count += 1
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {"raw": body.decode("utf-8", errors="replace")}

        status = 503 if self.server.post_count <= self.server.fail_first_posts else 200
        if self.server.response_delay_ms > 0:
            time.sleep(self.server.response_delay_ms / 1000.0)
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": "POST",
            "path": self.path,
            "status": status,
            "headers": {k: v for k, v in self.headers.items()},
            "payload": payload,
        }
        self._write_capture(entry)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = {
            "status": "ok" if status == 200 else "transient_failure",
            "post_count": self.server.post_count,
            "fail_first_posts": self.server.fail_first_posts,
        }
        self.wfile.write((json.dumps(body) + "\n").encode("utf-8"))

        self.server.request_count += 1
        if self.server.request_count >= self.server.max_requests:
            self.server.keep_running = False

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local webhook receiver with delay and transient failures.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795)
    parser.add_argument(
        "--output-json",
        default="research/breakthrough_lab/week24_controlled_rollout/week24_block2_live_webhook_capture.json",
    )
    parser.add_argument("--fail-first-posts", type=int, default=1)
    parser.add_argument("--response-delay-ms", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=12)
    args = parser.parse_args()

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    server = HTTPServer((args.host, args.port), _WebhookHandler)
    server.output_path = output_path
    server.fail_first_posts = max(0, int(args.fail_first_posts))
    server.response_delay_ms = max(0, int(args.response_delay_ms))
    server.max_requests = max(1, int(args.max_requests))
    server.request_count = 0
    server.post_count = 0
    server.health_count = 0
    server.keep_running = True
    server.capture = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": args.host,
            "port": int(args.port),
            "fail_first_posts": int(server.fail_first_posts),
            "response_delay_ms": int(server.response_delay_ms),
            "max_requests": int(server.max_requests),
        },
        "summary": {
            "requests_total": 0,
            "posts_total": 0,
            "health_checks_total": 0,
        },
        "requests": [],
    }

    print(f"Webhook receiver listening on http://{args.host}:{args.port}")
    print(f"Output JSON: {output_path}")

    while server.keep_running:
        server.handle_request()

    print("Webhook receiver closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
