import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    backend_base_url: str
    agent_host: str
    agent_port: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        backend_base_url=os.getenv("BACKEND_BASE_URL", "http://localhost:8080"),
        agent_host=os.getenv("AGENT_HOST", "0.0.0.0"),
        agent_port=int(os.getenv("AGENT_PORT", "8000")),
    )
