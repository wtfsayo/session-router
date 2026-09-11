"""Backend adapters. Each backend must report cache usage fields so the
router can keep the per-(session, model) warmth table accurate."""
from .base import Backend
from .local import OllamaBackend, LlamaCppBackend
from .cloud import OpenAICompatBackend, AnthropicBackend

__all__ = ["Backend", "OllamaBackend", "LlamaCppBackend",
           "OpenAICompatBackend", "AnthropicBackend"]
