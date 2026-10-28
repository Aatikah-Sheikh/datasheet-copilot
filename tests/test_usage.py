from types import SimpleNamespace

import pytest

from copilot.usage import BudgetExceeded, Limits, MeteredClient, UsageStore


class FakeClient:
    def __init__(self, prompt_tokens, completion_tokens):
        self.calls = 0
        usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: self._reply(usage)))

    def _reply(self, usage):
        self.calls += 1
        return SimpleNamespace(usage=usage, choices=[])


def test_counts_tokens_and_stops_at_session_cap():
    raw = FakeClient(60, 40)
    client = MeteredClient(raw, UsageStore(), Limits(per_session=200, per_day=10_000, total=10_000), {})
    client.chat.completions.create(model="m")
    client.chat.completions.create(model="m")
    assert client.session["tokens"] == 200
    with pytest.raises(BudgetExceeded, match="session"):
        client.chat.completions.create(model="m")
    assert raw.calls == 2  # the refused call never reached the API


def test_daily_and_total_caps_are_shared_across_sessions(tmp_path):
    store = UsageStore(tmp_path / "usage.json")
    limits = Limits(per_session=10_000, per_day=150, total=10_000)
    MeteredClient(FakeClient(100, 50), store, limits, {}).chat.completions.create(model="m")
    with pytest.raises(BudgetExceeded, match="today"):
        MeteredClient(FakeClient(1, 1), store, limits, {}).chat.completions.create(model="m")


def test_total_survives_restart(tmp_path):
    path = tmp_path / "usage.json"
    UsageStore(path).add(700, 300)
    reloaded = UsageStore(path)
    assert reloaded.snapshot()["total_tokens"] == 1000
    with pytest.raises(BudgetExceeded, match="all-time"):
        reloaded.check(Limits(per_session=10, per_day=10_000, total=1000))
