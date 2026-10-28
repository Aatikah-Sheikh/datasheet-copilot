"""Runtime settings, read from environment variables (a local .env file or Streamlit secrets)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Model IDs follow the Token Factory catalogue. Run `python scripts/check_setup.py`
# to list the Nemotron IDs your key can see and confirm these exist.
DEFAULT_REASONING_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_FAST_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


# Hard token caps (prompt + completion). Keep these well inside your free credits.
DEFAULT_MAX_TOKENS_PER_SESSION = 100_000
DEFAULT_MAX_TOKENS_PER_DAY = 300_000
DEFAULT_MAX_TOKENS_TOTAL = 1_000_000


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return int(value.replace("_", "")) if value else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    reasoning_model: str
    fast_model: str
    tavily_api_key: str | None
    max_tokens_per_session: int
    max_tokens_per_day: int
    max_tokens_total: int

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            api_key=_env("NEBIUS_API_KEY"),
            base_url=_env("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            reasoning_model=_env("NEBIUS_REASONING_MODEL", DEFAULT_REASONING_MODEL),
            fast_model=_env("NEBIUS_FAST_MODEL", DEFAULT_FAST_MODEL),
            tavily_api_key=_env("TAVILY_API_KEY"),
            max_tokens_per_session=_env_int("MAX_TOKENS_PER_SESSION", DEFAULT_MAX_TOKENS_PER_SESSION),
            max_tokens_per_day=_env_int("MAX_TOKENS_PER_DAY", DEFAULT_MAX_TOKENS_PER_DAY),
            max_tokens_total=_env_int("MAX_TOKENS_TOTAL", DEFAULT_MAX_TOKENS_TOTAL),
        )
