import time

from sessionrouter.cache import (cache_checkout_cost, keepalive_breakeven_seconds,
                                 p_hit, switch_cost, turn_cost, warm_tokens)
from sessionrouter.types import ModelCacheState, ModelSpec, Pricing


def spec(write=3.75, read=0.30, inp=3.0, ttl=300, min_tok=1024, cap=None):
    return ModelSpec("m", tier="cloud", backend="x",
                     pricing=Pricing(inp, 15.0, cached_input_per_mtok=read,
                                     cache_write_per_mtok=write,
                                     cache_ttl_seconds=ttl,
                                     cache_min_tokens=min_tok,
                                     cache_ttl_hard_cap=cap))


def test_p_hit_sliding_ttl():
    s = spec()
    st = ModelCacheState(last_seen_ts=1000.0, cached_prefix_tokens=5000,
                         observed_hit=True)
    assert p_hit(s, st, now=1000.0 + 100, prefix_tokens=5000) > 0.9
    assert p_hit(s, st, now=1000.0 + 400, prefix_tokens=5000) == 0.0  # TTL dead


def test_p_hit_hard_cap_decay():
    s = spec(cap=3600)
    st = ModelCacheState(last_seen_ts=0.0, cached_prefix_tokens=5000,
                         observed_hit=True)
    mid = p_hit(s, st, now=1800, prefix_tokens=5000)
    assert 0.0 < mid < 1.0
    assert p_hit(s, st, now=3601, prefix_tokens=5000) == 0.0


def test_p_hit_below_min_prefix():
    s = spec(min_tok=1024)
    st = ModelCacheState(last_seen_ts=0.0, cached_prefix_tokens=500,
                         observed_hit=True)
    assert p_hit(s, st, now=10, prefix_tokens=500) == 0.0


def test_switch_cost_positive_when_current_warm():
    cur, cand = spec(), spec(write=0.31, read=0.025, inp=0.25)
    now = 1000.0
    cur_st = ModelCacheState(last_seen_ts=now - 10, cached_prefix_tokens=40000,
                             observed_hit=True)
    sc = switch_cost(cur, cand, 40000, None, cur_st, now)
    # cold prefill 40k on cheap model vs warm read on expensive
    expected = 40000 * 0.31 / 1e6 - 40000 * 0.30 / 1e6
    assert abs(sc - expected) < 1e-9
    assert sc > 0


def test_turn_cost_uses_warm_prefix():
    s = spec()
    now = 1000.0
    st = ModelCacheState(last_seen_ts=now - 5, cached_prefix_tokens=40000,
                         observed_hit=True)
    c = turn_cost(s, history_tokens=40000, new_tokens=200,
                  est_output_tokens=500, cache_state=st, now=now)
    expected = (40000 * 0.30 + 200 * 3.0 + 500 * 15.0) / 1e6
    assert abs(c - expected) < 1e-9


def test_keepalive_breakeven():
    s = spec()  # write 3.75, read 0.30
    secs = keepalive_breakeven_seconds(s, ping_tokens=32, history_tokens=40000)
    # reprefill 0.15$ / ping ~0.012$ ≈ 12.5 windows × 300s ≈ 3750s
    assert 3000 < secs < 4500


def test_checkout_delta():
    assert abs(cache_checkout_cost(spec().pricing) - 3.45) < 1e-9
    s2 = spec(); s2.pricing.cache_write_per_mtok = None
    assert cache_checkout_cost(s2.pricing) == s2.pricing.input_per_mtok
