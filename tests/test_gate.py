from sessionrouter.gate import GateConfig, Verdict, check
from sessionrouter.types import BackendResponse, TokenLogprob, TokenUsage


def resp(text, lps=None, finish="stop", tools=None):
    return BackendResponse(text=text, model="m", finish_reason=finish,
                           tool_calls=tools or [],
                           logprobs=[TokenLogprob(t, lp) for t, lp in (lps or [])],
                           usage=TokenUsage(10, 10))


def msgs():
    from sessionrouter.types import Message
    return [Message(role="user", content="q")]


def test_truncated_escalates():
    v, d = check(resp("partial", finish="length"), msgs())
    assert v == Verdict.ESCALATE and d["reason"] == "truncated"


def test_refusal_escalates():
    v, d = check(resp("I cannot help with that."), msgs())
    assert v == Verdict.ESCALATE


def test_repetition_escalates():
    text = "the cat sat on the mat " * 40
    v, d = check(resp(text), msgs())
    assert v == Verdict.ESCALATE


def test_json_validation():
    v, _ = check(resp("{not json"), msgs(), GateConfig(expects_json=True))
    assert v == Verdict.ESCALATE
    v, _ = check(resp('{"a": 1}'), msgs(), GateConfig(expects_json=True))
    assert v == Verdict.ACCEPT


def test_bad_tool_args_escalate():
    tools = [{"function": {"arguments": "{bad json"}}]
    v, d = check(resp("", tools=tools), msgs())
    assert v == Verdict.ESCALATE and d["reason"] == "invalid_tool_args"


def test_low_confidence_escalates():
    lps = [("a", -0.1)] * 20 + [("42", -7.0), ("?", -5.0)]
    text = "unique words here " + " ".join(f"w{i}" for i in range(20)) + " 42 ?"
    v, d = check(resp(text, lps=lps), msgs())
    assert v == Verdict.ESCALATE and "logprob" in d["reason"]


def test_confident_accepts():
    lps = [(f"t{i}", -0.2) for i in range(30)]
    text = " ".join(f"t{i}" for i in range(30))
    v, d = check(resp(text, lps=lps), msgs())
    assert v == Verdict.ACCEPT
