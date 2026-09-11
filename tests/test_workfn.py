from sessionrouter.workfn import WorkFunction, hitting_cost


def test_wfa_stays_when_switch_not_worth_it():
    # two models; model 1 slightly cheaper per turn but expensive to enter
    wf = WorkFunction(2)
    d = [[0, 1.0], [1.0, 0]]
    # model1 cheaper by 0.1/turn; switch cost 1.0 => needs ~10 turns to pay off
    for _ in range(3):
        s = wf.step([0.5, 0.4], d)
        assert s == 0  # stays: cumulative saving 0.3 < switch cost 1.0


def test_wfa_switches_when_breakeven_crossed():
    wf = WorkFunction(2)
    d = [[0, 1.0], [1.0, 0]]
    last = 0
    for _ in range(30):
        last = wf.step([0.5, 0.4], d)
    assert last == 1  # eventually switches once cumulative savings exceed cost


def test_wfa_holds_against_one_turn_spike():
    wf = WorkFunction(2)
    d = [[0, 2.0], [2.0, 0]]
    wf.step([1.0, 0.2], d)          # turn 1: model1 better
    s = wf.step([0.9, 0.2], d)      # turn 2: still cheaper long-run? holds
    assert s == 0                   # switch cost 2.0 > savings


def test_hitting_cost_math():
    c = hitting_cost(price_per_mtok=3.0, tokens=10000,
                     quality_deficit=0.5, quality_weight=2.0)
    assert abs(c - (0.03 + 1.0)) < 1e-9
