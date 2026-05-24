"""Centralized settings, loaded from env vars (or `.env`)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider — one of ``ollama`` (default, fully-local), ``openai``,
    # ``anthropic``, ``stub`` (deterministic mock for tests).
    llm_backend: str = Field("ollama", alias="REGPILOT_LLM")

    # Ollama
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    chat_model: str = Field("qwen2.5:3b-instruct", alias="REGPILOT_CHAT_MODEL")
    embed_model: str = Field("nomic-embed-text", alias="REGPILOT_EMBED_MODEL")
    ollama_timeout_s: float = Field(30.0, alias="OLLAMA_TIMEOUT_S")

    # OpenAI (hosted) — used when REGPILOT_LLM=openai
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_base_url: str = Field("", alias="OPENAI_BASE_URL")  # blank → default
    openai_chat_model: str = Field("gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embed_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBED_MODEL")

    # Anthropic (hosted) — used when REGPILOT_LLM=anthropic. Anthropic has no
    # embedding API, so embeddings fall back to Ollama when this backend is
    # active. Set ``OLLAMA_BASE_URL`` accordingly.
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_chat_model: str = Field("claude-3-5-haiku-latest", alias="ANTHROPIC_CHAT_MODEL")

    embed_parallelism: int = Field(8, alias="REGPILOT_EMBED_PARALLELISM")

    # LLM-primary toggles. Defaults to LLM-driven for proper agentic behaviour;
    # set any to ``true`` to opt back into the deterministic fast-path
    # (useful for CPU-only Ollama installs that need < 30s latency).
    intake_fast: bool = Field(False, alias="REGPILOT_INTAKE_FAST")
    rerank_fast: bool = Field(False, alias="REGPILOT_RERANK_FAST")
    synth_fast: bool = Field(False, alias="REGPILOT_SYNTH_FAST")

    # Semantic-similarity classifier threshold. Annex III area is considered a
    # match if cosine similarity ≥ this value. 0.45 was the proven-stable
    # default across the main 16-question gold set; a lower 0.35 was tried
    # to surface more candidates for the LLM but it regressed limited-risk
    # classification on main (chatbots / deepfakes), so 0.45 stays the
    # default. The enriched canonical examples in the classifier already
    # raise the cosine scores enough to catch paraphrases like "traffic
    # light timing" or "PhD applicant ranking" without lowering the bar.
    semantic_match_threshold: float = Field(0.45, alias="REGPILOT_SEM_THRESHOLD")

    chroma_dir: Path = Field(REPO_ROOT / "data" / "chroma", alias="REGPILOT_CHROMA_DIR")
    data_dir: Path = Field(REPO_ROOT / "data" / "raw", alias="REGPILOT_DATA_DIR")

    log_level: str = Field("INFO", alias="REGPILOT_LOG_LEVEL")

    # EU AI Act source. Publications Office cellar URL with content negotiation —
    # EUR-Lex sits behind a CloudFront WAF that 403s scripted clients, so we go
    # straight to the publications.europa.eu resource which serves the PDF when
    # we send `Accept: application/pdf` + `Accept-Language: eng`.
    ai_act_celex: str = "32024R1689"
    ai_act_pdf_url: str = (
        "https://publications.europa.eu/resource/cellar/"
        "dc8116a1-3fe6-11ef-865a-01aa75ed71a1"
    )

    # Retrieval defaults — top_k_rerank must be >= max tier priority list size
    # (high-risk has 12 obligation Articles) so the diversified pre-seed in
    # the rerank node can surface one chunk per Article without budget overflow.
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_rerank: int = 12
    rrf_k: int = 60

    # Validator loop cap
    max_validator_loops: int = 2

    # Production guard: LangGraph will raise GraphRecursionError after this
    # many node hops. Sized at ~2× the worst-case path (intake → triage →
    # rag → mapper → synth → validator → mapper → synth → validator).
    graph_recursion_limit: int = 40

    # State durability. ``memory`` (default) = ephemeral; ``sqlite`` = on-disk
    # checkpointer keyed by ``thread_id`` so a crashed Streamlit / Docker
    # restart can resume the last in-flight run.
    checkpointer: str = Field("memory", alias="REGPILOT_CHECKPOINTER")
    checkpoint_path: Path = Field(
        REPO_ROOT / "data" / "checkpoints.sqlite",
        alias="REGPILOT_CHECKPOINT_PATH",
    )

    # Structured JSON logs (off by default for human-readable local dev,
    # on inside docker-compose for log shippers).
    log_json: bool = Field(False, alias="REGPILOT_LOG_JSON")

    @property
    def is_stub(self) -> bool:
        return self.llm_backend.lower() == "stub"

    @property
    def provider(self) -> str:
        """Normalised provider name (``ollama`` / ``openai`` / ``anthropic`` / ``stub``)."""

        return self.llm_backend.lower()


settings = Settings()  # type: ignore[call-arg]  # pydantic-settings reads defaults from env
