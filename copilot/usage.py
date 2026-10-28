"""Token metering and hard caps on Token Factory spend.

Every chat completion goes through `MeteredClient`, which refuses the call once a
per-session, per-day or all-time token cap is reached. Day and all-time counters are
shared by every session in the process and persisted to a JSON file, so they survive
a restart. Caps are checked before each call, so one call can overshoot by at most its
own size (bounded by max_tokens plus the prompt).
"""

import datetime
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    per_session: int
    per_day: int
    total: int


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


class UsageStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        data = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "day": _today(), "day_tokens": 0}
        if self.path and self.path.exists():
            try:
                data.update(json.loads(self.path.read_text()))
            except (OSError, json.JSONDecodeError):
                pass
        return data

    def _roll_day(self) -> None:
        if self._data["day"] != _today():
            self._data["day"] = _today()
            self._data["day_tokens"] = 0

    def snapshot(self) -> dict:
        with self._lock:
            self._roll_day()
            return dict(self._data)

    def check(self, limits: Limits) -> None:
        with self._lock:
            self._roll_day()
            if self._data["total_tokens"] >= limits.total:
                raise BudgetExceeded("The app has reached its all-time token cap, so new requests are paused.")
            if self._data["day_tokens"] >= limits.per_day:
                raise BudgetExceeded("The app has reached today's token cap. It resets at 00:00 UTC.")

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._roll_day()
            used = prompt_tokens + completion_tokens
            self._data["prompt_tokens"] += prompt_tokens
            self._data["completion_tokens"] += completion_tokens
            self._data["total_tokens"] += used
            self._data["day_tokens"] += used
            self._data["calls"] += 1
            if self.path:
                try:
                    self.path.write_text(json.dumps(self._data))
                except OSError:
                    pass


class MeteredClient:
    """Wraps an OpenAI-compatible client; exposes `chat.completions.create` with caps applied."""

    def __init__(self, client, store: UsageStore, limits: Limits, session: dict) -> None:
        self._client = client
        self.store = store
        self.limits = limits
        self.session = session  # mutable per-session counter: {"tokens": int}
        self.session.setdefault("tokens", 0)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def check(self) -> None:
        if self.session["tokens"] >= self.limits.per_session:
            raise BudgetExceeded("This session has used its token allowance. Clear the conversation and reload to start a new one.")
        self.store.check(self.limits)

    def _create(self, **kwargs):
        self.check()
        response = self._client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.session["tokens"] += prompt + completion
        self.store.add(prompt, completion)
        return response
