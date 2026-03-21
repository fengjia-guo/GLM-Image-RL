#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List


def parse_args():
    parser = argparse.ArgumentParser(description="HPSv3 reward service")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def to_scalar(score_item: Any) -> float:
    if isinstance(score_item, (list, tuple)) and score_item:
        return to_scalar(score_item[0])
    try:
        import torch

        if torch.is_tensor(score_item):
            return float(score_item.detach().cpu().flatten()[0].item())
    except Exception:
        pass
    if isinstance(score_item, dict):
        for key in ("mu", "score", "reward"):
            if key in score_item:
                return float(score_item[key])
    return float(score_item)


class HPSv3Service:
    def __init__(self, device: str):
        from hpsv3 import HPSv3RewardInferencer

        self.inferencer = HPSv3RewardInferencer(device=device)

    def score(self, prompts: List[str], image_paths: List[str]) -> List[float]:
        if hasattr(self.inferencer, "reward"):
            fn = self.inferencer.reward
        elif hasattr(self.inferencer, "score"):
            fn = self.inferencer.score
        else:
            raise AttributeError("hpsv3 inferencer has neither `reward` nor `score`.")

        errors = []
        call_patterns = [
            lambda: fn(image_paths, prompts),
            lambda: fn(prompts, image_paths=image_paths),
            lambda: fn(prompts, image_paths),
        ]
        raw_scores = None
        for call in call_patterns:
            try:
                raw_scores = call()
                break
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        if raw_scores is None:
            raise RuntimeError(
                "Failed to call hpsv3 inferencer with known signatures. "
                + " | ".join(errors)
            )
        return [to_scalar(item) for item in raw_scores]


def make_handler(service: HPSv3Service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict):
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True})
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/score":
                self._send(404, {"ok": False, "error": "not found"})
                return
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                prompts = payload["prompts"]
                images_b64 = payload["images_base64"]
                if len(prompts) != len(images_b64):
                    raise ValueError("prompts/images_base64 length mismatch")

                with tempfile.TemporaryDirectory(prefix="hpsv3_srv_") as tmpdir:
                    image_paths: List[str] = []
                    for idx, b64 in enumerate(images_b64):
                        raw = base64.b64decode(b64)
                        path = os.path.join(tmpdir, f"sample_{idx:06d}.png")
                        with open(path, "wb") as f:
                            f.write(raw)
                        image_paths.append(path)
                    scores = service.score(prompts=prompts, image_paths=image_paths)
                self._send(200, {"ok": True, "scores": scores})
            except Exception as e:
                self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    args = parse_args()
    service = HPSv3Service(device=args.device)
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"hpsv3 reward service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
