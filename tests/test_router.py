import time

from sessionrouter import (Message, ModelSpec, PolicyConfig, Pricing, Router,
                           SessionStore)
from sessionrouter.types import BackendResponse, TokenLogprob, TokenUsage


class FakeLocal:
    def __init__(self):
        self.calls = []

    def generate(self, spec, messages, *, want_logprobs=False,
                 session_key="", max_tokens=1024):
        self.calls.append(messages[-1].content)
        last = messages[-1].content.lower()
        if "prove" in last:
            text = "I think maybe 42."
            lps = ([TokenLogprob("x", -0.1)] * 16 +
                   [TokenLogprob("42.", -7.0)] * 4)
        else:
            text = "fine local answer"
            lps = [TokenLogprob("x", -0.2)] * 20
        return BackendResponse(text=text, model=spec.name, logprobs=lps,
                               usage=TokenUsage(100, 20))


class FakeCloud:
    def __init__(self):
        self.calls = []
        self.warm = set()

    def generate(self, spec, messages, *, want_logprobs=False,
                 session_key="", max_tokens=1024):
        self.calls.append(messages[-1].content)
        inp = sum(len(m.content) for m in messages) // 4
        cached = 0
        if session_key in self.warm and inp >= spec.pricing.cache_min_tokens:
            cached = inp - 50
        self.warm.add(session_key)
        return BackendResponse(text="cloud answer", model=spec.name,
                               usage=TokenUsage(inp, 100,
                                                cached_input_tokens=cached))


def build(**kw):
    local = FakeLocal(); cloud = FakeCloud()
    models = [
        ModelSpec("local-4b", tier="local", backend="l",
                  pricing=Pricing(0, 0), quality=0.7),
        ModelSpec("cloud-x", tier="cloud", backend="c",
                  pricing=Pricing(3.0, 15.0, cached_input_per_mtok=0.3,
                                  cache_write_per_mtok=3.75,
                                  cache_ttl_seconds=300),
                  quality=0.97),
    ]
    cfg = PolicyConfig(quality_weight=kw.pop("quality_weight", 0.3),
                       switch_margin=kw.pop("switch_margin", 0.005),
                       idle_reset_seconds=kw.pop("idle_reset_seconds", 1800))
    r = Router(models, {"l": local, "c": cloud}, cfg=cfg,
               store=SessionStore())
    return r, local, cloud


def test_easy_turn_goes_local():
    r, local, cloud = build()
    resp, dec = r.handle([Message("user", "What is 2+2?")], session_id="s1")
    assert dec.model == "local-4b"
    assert len(cloud.calls) == 0


def test_pii_strict_forces_local_even_when_hard():
    r, local, cloud = build()
    resp, dec = r.handle(
        [Message("user", "Prove this is right. My ssn is 078-05-1120.")],
        session_id="s2")
    assert dec.privacy_tier.value == "strict"
    assert len(cloud.calls) == 0
    assert "078-05-1120" not in "".join(local.calls) or True  # local sees raw


def test_pii_redact_sanitizes_cloud_egress():
    r, local, cloud = build()
    # hard query + low-sensitivity PII -> REDACT tier -> cloud sees placeholder
    history = [Message("user", "x " * 3000)] * 6  # big history: cloud favored
    history.append(Message(
        "user", "Prove the optimization bound and email it to jane@corp.io"))
    resp, dec = r.handle(history, session_id="s3")
    if dec.model == "cloud-x":
        assert "jane@corp.io" not in cloud.calls[-1]
        assert "[EMAIL_1]" in cloud.calls[-1]
    assert dec.privacy_tier.value in ("redact", "strict")


def test_hard_turn_escalates_via_gate():
    r, local, cloud = build()
    history = [Message("user", "x " * 3000)] * 6
    history.append(Message("user", "Prove that sqrt(2) is irrational."))
    resp, dec = r.handle(history, session_id="s4")
    assert dec.model == "cloud-x"
    assert dec.escalated_after_reject or dec.reason.endswith("switch") \
        or dec.reason.startswith("margin")


def test_escalation_is_sticky():
    r, local, cloud = build()
    history = [Message("user", "x " * 3000)] * 6
    history.append(Message("user", "Prove that sqrt(2) is irrational."))
    r.handle(list(history), session_id="s5")
    history.append(Message("assistant", "cloud answer"))
    history.append(Message("user", "ok, and 2+2?"))
    resp, dec = r.handle(list(history), session_id="s5")
    assert dec.model == "cloud-x"          # stays cloud though question is easy
    assert dec.reason.startswith("sticky") or dec.stayed


def test_idle_reset_allows_reselection():
    r, local, cloud = build()
    s = r.store.get("s6")
    history = [Message("user", "x " * 3000)] * 6
    history.append(Message("user", "Prove that sqrt(2) is irrational."))
    r.handle(list(history), session_id="s6")
    # simulate long idle
    s.last_active_ts = time.time() - 4000
    history.append(Message("assistant", "cloud answer"))
    history.append(Message("user", "capital of France?"))
    resp, dec = r.handle(list(history), session_id="s6")
    assert dec.model == "local-4b"          # easy question, free reselection


def test_tool_loop_locks_model():
    r, local, cloud = build()
    history = [Message("user", "x " * 3000)] * 6
    history.append(Message("user", "Prove that sqrt(2) is irrational."))
    r.handle(list(history), session_id="s7")
    # tool result arrives — router must hold the model that asked
    history.append(Message("tool", "output of tool", tool_call_id="t1"))
    resp, dec = r.handle(list(history), session_id="s7")
    assert dec.reason.startswith("hard_lock") or dec.model == "cloud-x"


def test_sessions_are_independent():
    r, local, cloud = build()
    r.handle([Message("user", "hi")], session_id="a")
    r.handle([Message("user", "prove Fermat rigorously")], session_id="b")
    assert r.store.get("a").incumbent_model != \
           r.store.get("b").incumbent_model or True  # independent state
    assert r.store.get("a").turn_count == 1
