from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production RAG Platform"
    environment: str = "development"

    database_url: str = (
        "postgresql+psycopg://rag_user:rag_password"
        "@localhost:5432/rag_db"
    )

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-mini"
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()