from crucible.metrics import MetricsLedger, UsageRecord, cost_usd


def test_cost_usd_uses_price_table():
    # 1M input + 1M output of flash at (0.30, 2.50)
    assert cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000) == round(0.30 + 2.50, 6)


def test_cost_zero_for_offline_model():
    assert cost_usd("scripted-offline", 1000, 1000) == 0.0


def test_usage_record_derived_fields():
    r = UsageRecord("researcher", "gemini-2.5-flash", prompt_tokens=100, output_tokens=50, latency_s=2.0)
    assert r.total_tokens == 150
    assert r.tokens_per_sec == 25.0
    assert r.cost > 0


def test_ledger_aggregates_and_percentiles():
    ledger = MetricsLedger()
    ledger.add(UsageRecord("researcher", "gemini-2.5-flash", 100, 40, 1.0))
    ledger.add(UsageRecord("reviser", "gemini-2.5-flash", 80, 30, 2.0))
    ledger.add(UsageRecord("judge", "gemini-2.5-flash", 120, 20, 3.0))

    s = ledger.summary()
    assert s["calls"] == 3
    assert s["total_tokens"] == 100 + 40 + 80 + 30 + 120 + 20
    assert s["total_cost_usd"] > 0
    assert s["p50_latency_s"] == 2.0
    assert s["cost_per_request_usd"] == round(s["total_cost_usd"] / 3, 6)

    per_agent = ledger.per_agent()
    assert set(per_agent) == {"researcher", "reviser", "judge"}
    assert per_agent["researcher"]["calls"] == 1
