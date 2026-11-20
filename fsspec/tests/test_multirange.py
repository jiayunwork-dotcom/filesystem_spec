import random

import pytest

from fsspec.multirange import (
    CoalesceBudget,
    RangeResponseError,
    plan_range_requests,
    validate_cat_ranges_args,
    validate_range_policy,
)

POLICY = {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": 50}


def test_validate_policy_ok():
    assert validate_range_policy(POLICY) == (4, 100, 50)
    # integer-valued subclasses other than bool are accepted
    assert validate_range_policy(
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": 0}
    ) == (4, 100, 0)


@pytest.mark.parametrize(
    "policy",
    [
        None.__class__ and "notadict",
        ["max_ranges"],
        {"max_ranges": 4, "max_request_bytes": 100},  # missing extra_bytes
        {"max_ranges": 4, "extra_bytes": 0},  # missing max_request_bytes
        {"extra_bytes": 0},  # missing two
        {},  # missing all
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": 0, "foo": 1},
        {"max_ranges": True, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": False, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": True},
        {"max_ranges": 0, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": -1, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": 0, "extra_bytes": 0},
        {"max_ranges": 4, "max_request_bytes": 100, "extra_bytes": -1},
        {"max_ranges": 4.5, "max_request_bytes": 100, "extra_bytes": 0},
        {"max_ranges": "4", "max_request_bytes": 100, "extra_bytes": 0},
    ],
)
def test_validate_policy_bad(policy):
    with pytest.raises(ValueError):
        validate_range_policy(policy)


def test_validate_args_ok():
    out = validate_cat_ranges_args(["a", "b"], [0, 5], [5, 5])
    assert out == [(0, "a", 0, 5), (1, "b", 5, 5)]


@pytest.mark.parametrize(
    "paths,starts,ends,exc",
    [
        ("notalist", [0], [1], TypeError),
        (["a"], 0, [1], ValueError),  # starts not a list
        (["a"], [0], 1, ValueError),  # ends not a list
        (["a", "b"], [0], [1], ValueError),  # length mismatch
        (["a"], [0], [1, 2], ValueError),
        (["a"], [-1], [1], ValueError),  # negative start
        (["a"], [0], [-1], ValueError),  # negative end
        (["a"], [5], [4], ValueError),  # start > end
        (["a"], [True], [1], ValueError),  # bool start
        (["a"], [0], [False], ValueError),  # bool end
        (["a"], [0.5], [1], ValueError),  # float
        (["a"], ["0"], [1], ValueError),  # str
    ],
)
def test_validate_args_bad(paths, starts, ends, exc):
    with pytest.raises(exc):
        validate_cat_ranges_args(paths, starts, ends)


def test_plan_empty():
    plan = plan_range_requests([], 4, 100, 50)
    assert plan.requests == []
    assert plan.input_map == []
    assert plan.union_bytes == 0
    assert plan.declared_bytes == 0
    assert plan.budget.remaining == 50


def test_plan_dense_coalescing():
    # 256 x 64B blocks at stride 80, as in the feature benchmarks
    items = [("u", 80 * j, 80 * j + 64) for j in range(256)]
    plan = plan_range_requests(items, 8, 4096, 4096)
    assert len(plan.requests) <= 8
    assert plan.union_bytes == 256 * 64
    assert plan.declared_bytes <= plan.union_bytes + 4096
    for _, ivs in plan.requests:
        assert len(ivs) <= 8
        assert sum(e - s for s, e in ivs) <= 4096


def test_plan_sparse_no_extra():
    items = [("u", 1024 * j, 1024 * j + 64) for j in range(256)]
    plan = plan_range_requests(items, 8, 4096, 0)
    assert len(plan.requests) == 32
    assert plan.declared_bytes == plan.union_bytes == 256 * 64


def test_plan_urls_never_mix():
    items = [("a", 0, 10), ("b", 0, 10), ("a?x=1", 0, 10), ("a", 10, 20)]
    plan = plan_range_requests(items, 100, 1000, 1000)
    urls = {url for url, _ in plan.requests}
    # the query string is part of the URL identity
    assert urls == {"a", "b", "a?x=1"}
    for url, ivs in plan.requests:
        assert all(u == url for u, _ in plan.requests if u == url)


def test_plan_split_long_range():
    plan = plan_range_requests([("u", 0, 10000)], 8, 4096, 0)
    assert [ivs for _, ivs in plan.requests] == [
        ((0, 4096),),
        ((4096, 8192),),
        ((8192, 10000),),
    ]
    assert plan.input_map == [(0, 1, 2)]
    assert plan.declared_bytes == plan.union_bytes == 10000


def test_plan_duplicates_and_contained():
    items = [("u", 0, 100), ("u", 0, 100), ("u", 20, 50), ("u", 90, 110)]
    plan = plan_range_requests(items, 4, 1000, 0)
    assert plan.union_bytes == 110
    assert plan.declared_bytes == 110
    assert len(plan.requests) == 1
    # every input maps to the one request
    assert plan.input_map == [(0,), (0,), (0,), (0,)]


def check_plan_invariants(items, max_ranges, max_request_bytes, extra_bytes):
    plan = plan_range_requests(items, max_ranges, max_request_bytes, extra_bytes)
    per_url = {}
    for url, ivs in plan.requests:
        assert len(ivs) <= max_ranges
        assert sum(e - s for s, e in ivs) <= max_request_bytes
        for (s1, e1), (s2, e2) in zip(ivs, ivs[1:]):
            assert s1 < e1 <= s2 < e2  # sorted, disjoint, non-empty
        per_url.setdefault(url, []).extend(ivs)
    # declared intervals per URL are disjoint and cover the union exactly
    union = {}
    for url, s, e in items:
        union.setdefault(url, []).append((s, e))
    total_union = 0
    for url, pairs in union.items():
        merged = _merge(pairs)
        total_union += sum(e - s for s, e in merged)
        declared = sorted(per_url[url])
        # declared intervals never overlap (no byte is downloaded twice)
        for (s1, e1), (s2, e2) in zip(declared, declared[1:]):
            assert e1 <= s2
        # and they cover the union (bridged gaps make them a superset)
        dm = _merge(declared)
        j = 0
        for s, e in merged:
            while j < len(dm) and dm[j][1] < e:
                j += 1
            assert j < len(dm) and dm[j][0] <= s
    assert plan.union_bytes == total_union
    assert plan.declared_bytes <= plan.union_bytes + extra_bytes
    assert plan.declared_bytes >= plan.union_bytes
    # input map: the listed requests really do cover each input, in order
    for (url, s, e), reqs in zip(items, plan.input_map):
        assert reqs
        segs = []
        for r in reqs:
            segs.extend(plan.requests[r][1])
        segs = sorted(set(segs))
        # walk coverage
        pos = s
        for ss, ee in segs:
            if ee <= pos:
                continue
            assert ss <= pos, (url, s, e, reqs)
            pos = ee
            if pos >= e:
                break
        assert pos >= e
        # request indices ascending without duplicates
        assert list(reqs) == sorted(set(reqs))
    return plan


def _merge(ivs):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def test_plan_fuzz_invariants():
    rng = random.Random(20240501)
    for trial in range(300):
        nurls = rng.randint(1, 3)
        n = rng.randint(1, 60)
        items = []
        for _ in range(n):
            url = f"u{rng.randrange(nurls)}"
            s = rng.randrange(0, 5000)
            e = s + rng.randrange(1, 300)
            items.append((url, s, e))
        max_ranges = rng.randint(1, 8)
        max_request_bytes = rng.randint(50, 2000)
        extra_bytes = rng.randint(0, 500)
        check_plan_invariants(items, max_ranges, max_request_bytes, extra_bytes)


def test_plan_fuzz_with_long_ranges():
    rng = random.Random(99)
    for _ in range(50):
        items = []
        for _ in range(rng.randint(1, 10)):
            s = rng.randrange(0, 10000)
            items.append(("u", s, s + rng.randrange(1, 9000)))
        check_plan_invariants(items, rng.randint(1, 4), rng.randint(100, 3000), 100)


# --------------------------------------------------------------------------
# the single call-level CoalesceBudget shared across backends/batches
# --------------------------------------------------------------------------


def test_budget_take_and_overrun():
    budget = CoalesceBudget(5)
    budget.take(2)
    assert budget.remaining == 3
    budget.take(3)
    assert budget.remaining == 0
    with pytest.raises(RangeResponseError):
        budget.take(1)
    # a rejected charge does not mutate the budget
    assert budget.remaining == 0


def test_plan_records_planner_spend():
    plan = plan_range_requests([("u", 0, 40), ("u", 60, 100)], 1, 1000, 20)
    # the only way to satisfy max_ranges=1 is bridging the 20-byte gap
    assert plan.planner_spend == 20
    assert plan.declared_bytes == plan.union_bytes + 20
    assert plan.budget.remaining == 0


def test_shared_budget_between_two_planners():
    # two independent backends plan sequentially against one budget:
    # the second planner must see what the first one spent
    budget = CoalesceBudget(30)
    plan1 = plan_range_requests(
        [("http://a/x", 0, 40), ("http://a/x", 60, 100)],
        1,
        1000,
        30,
        budget=budget,
    )
    assert plan1.planner_spend == 20
    assert plan1.budget is budget
    assert budget.remaining == 10

    # the second backend's 15-byte gap cannot be bridged with 10 left,
    # so its ranges stay split instead of overspending
    plan2 = plan_range_requests(
        [("https://b/y", 0, 10), ("https://b/y", 25, 35)],
        1,
        1000,
        30,
        budget=budget,
    )
    assert plan2.planner_spend == 0
    assert len(plan2.requests) == 2
    assert budget.remaining == 10


def test_shared_budget_zero_carries_over():
    budget = CoalesceBudget(0)
    plan = plan_range_requests(
        [("u", 0, 10), ("u", 100, 110)], 1, 1000, 0, budget=budget
    )
    assert plan.planner_spend == 0
    assert len(plan.requests) == 2
    assert budget.remaining == 0
    # no slack left for a server-coalesced response either
    with pytest.raises(RangeResponseError):
        budget.take(1)


def test_plan_creates_own_budget_by_default():
    plan = plan_range_requests([("u", 0, 10)], 1, 1000, 7)
    assert isinstance(plan.budget, CoalesceBudget)
    assert plan.budget.remaining == 7


def test_plan_budget_field_present_in_empty_plan():
    plan = plan_range_requests([], 4, 100, 50)
    assert plan.planner_spend == 0
    assert isinstance(plan.budget, CoalesceBudget)
    assert plan.budget.remaining == 50
