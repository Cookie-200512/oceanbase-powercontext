"""Cross-call embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

_DEFAULT_MAX_BATCH_SIZE = 64
_DEFAULT_COALESCE_MS = 5.0
_DEFAULT_CACHE_SIZE = 1024


class CoalescingEmbeddingModel:
    """Probe build: forwards to the wrapped model without any scheduling."""

    def __init__(
        self,
        model,
        *,
        max_batch_size=_DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms=_DEFAULT_COALESCE_MS,
        cache_size=_DEFAULT_CACHE_SIZE,
    ):
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size
        self._cache = {}

    async def embed(self, texts):
        items = tuple(texts)
        if not items:
            from powercontext.builtin.inference import EmbeddingResult, InferenceUsage

            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        return await self._model.embed(items)


__all__ = ["CoalescingEmbeddingModel"]
