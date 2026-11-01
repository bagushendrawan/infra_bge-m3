from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BGE_M3_", env_file=".env", extra="ignore")
    model: str = "BAAI/bge-m3"
    device: str = Field(default="cuda:0", pattern=r"^(cpu|cuda(:[0-9]+)?)$")
    environment: str = "local"
    batch_size: int = Field(default=4, gt=0)
    startup_timeout: float = Field(default=600, gt=0)
    timeout: float = Field(default=120, gt=0)
    max_items: int = Field(default=16, gt=0)
    max_text_chars: int = Field(default=32768, gt=0)
    max_total_chars: int = Field(default=131072, gt=0)
    max_result_bytes: int = Field(default=67108864, gt=0)
