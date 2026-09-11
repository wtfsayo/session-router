"""CLI: `session-router serve` / `session-router chat` / `session-router demo`."""
from __future__ import annotations

import argparse
import json
import sys

from .backends import (AnthropicBackend, LlamaCppBackend, OllamaBackend,
                       OpenAICompatBackend)
from .policy import PolicyConfig
from .router import Router
from .trace import TraceLog
from .types import Message, ModelSpec, Pricing


def default_router(args) -> Router:
    models = [
        ModelSpec(name=args.local_model, tier="local",
                  backend=args.local_backend,
                  pricing=Pricing(0.0, 0.0), quality=0.72,
                  max_context=args.local_ctx),
        ModelSpec(name=args.cloud_model, tier="cloud",
                  backend=args.cloud_backend,
                  pricing=Pricing(
                      input_per_mtok=args.cloud_in,
                      output_per_mtok=args.cloud_out,
                      cached_input_per_mtok=args.cloud_cached,
                      cache_write_per_mtok=args.cloud_write,
                      cache_ttl_seconds=300.0),
                  quality=0.97, max_context=200000),
    ]
    backends = {
        "ollama": OllamaBackend(keep_alive=args.keep_alive),
        "llamacpp": LlamaCppBackend(),
        "openai_compat": OpenAICompatBackend(
            base_url=args.cloud_base, api_key=args.cloud_key),
        "anthropic": AnthropicBackend(api_key=args.anthropic_key),
    }
    scorer = None
    if args.scorer:
        from .trained import load_scorer
        scorer = load_scorer(args.scorer)
    return Router(models, backends,
                  cfg=PolicyConfig(decision_mode=args.mode),
                  scorer=scorer,
                  tracer=TraceLog(args.trace_db))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="session-router")
    p.add_argument("--local-model", default="qwen3:4b")
    p.add_argument("--local-backend", default="ollama",
                   choices=["ollama", "llamacpp"])
    p.add_argument("--local-ctx", type=int, default=8192)
    p.add_argument("--keep-alive", default="10m")
    p.add_argument("--cloud-model", default="gpt-5-mini")
    p.add_argument("--cloud-backend", default="openai_compat",
                   choices=["openai_compat", "anthropic"])
    p.add_argument("--cloud-base", default="https://api.openai.com/v1")
    p.add_argument("--cloud-key", default="")
    p.add_argument("--anthropic-key", default="")
    p.add_argument("--cloud-in", type=float, default=0.25)
    p.add_argument("--cloud-out", type=float, default=2.0)
    p.add_argument("--cloud-cached", type=float, default=0.025)
    p.add_argument("--cloud-write", type=float, default=0.31)
    p.add_argument("--mode", default="margin", choices=["margin", "wfa"])
    p.add_argument("--scorer", default="",
                   help="path to a trained scorer artifact "
                        "(bench/train_scorer.py)")
    p.add_argument("--trace-db", default="router_trace.db")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8400)

    c = sub.add_parser("chat")
    c.add_argument("text")
    c.add_argument("--session", default="cli")
    c.add_argument("--history", default="[]",
                   help="JSON list of prior messages")

    sub.add_parser("demo")
    args = p.parse_args(argv)

    if args.cmd == "serve":
        from .server import serve
        serve(default_router(args), args.host, args.port)
        return 0
    if args.cmd == "chat":
        r = default_router(args)
        hist = [Message(**m) for m in json.loads(args.history)]
        hist.append(Message(role="user", content=args.text))
        resp, dec = r.handle(hist, session_id=args.session)
        print(f"[{dec.model} | {dec.reason} | {dec.privacy_tier.value}]")
        print(resp.text)
        return 0
    if args.cmd == "demo":
        from .demo import run_demo
        run_demo()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
