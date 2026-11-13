"""Cross-call embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference import EmbeddingResult, InferenceUsage

_DEFAULT_MAX_BATCH_SIZE = 64
_DEFAULT_COALESCE_MS = 5.0
_DEFAULT_CACHE_SIZE = 1024


class _Slot:
    """One pending text shared by every caller that asked for it."""

    __slots__ = ("event", "vector", "error", "usage")

    def __init__(self):
        self.event = asyncio.Event()
        self.vector = None
        self.error = None
        self.usage = None


class CoalescingEmbeddingModel:
    """Share in-flight embedding work and cache successful vectors between calls."""

    def __init__(
        self,
        model,
        *,
        max_batch_size=_DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms=_DEFAULT_COALESCE_MS,
        cache_size=_DEFAULT_CACHE_SIZE,
    ):
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache = OrderedDict()
        self._slots = {}
        self._pending = []
        self._flushing = False

    async def embed(self, texts):
        """Embed texts in order, sharing duplicate and concurrent work."""
        items = tuple(texts)
        if not items:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(items))
        vectors = {}
        slots = []
        start_flush = False

        for text in unique:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            slot = self._slots.get(text)
            if slot is None:
                slot = _Slot()
                self._slots[text] = slot
                self._pending.append(text)
                if not self._flushing:
                    self._flushing = True
                    start_flush = True
            slots.append((text, slot))

        if start_flush:
            asyncio.ensure_future(self._flush())

        usage = None
        if slots:
            await asyncio.gather(*(slot.event.wait() for _, slot in slots))
            batches = {}
            for text, slot in slots:
                if slot.error is not None:
                    raise slot.error
                vectors[text] = slot.vector
                if slot.usage is not None:
                    batches[id(slot.usage)] = slot.usage
            usage = _sum_usage(tuple(batches.values()))
        if usage is None:
            usage = InferenceUsage(requests=0)

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in items),
            usage=usage,
        )

    async def _flush(self):
        try:
            if self._coalesce_seconds > 0:
                await asyncio.sleep(self._coalesce_seconds)
            pending = self._pending
            self._pending = []
            self._flushing = False
            for start in range(0, len(pending), self._max_batch_size):
                await self._run_batch(tuple(pending[start : start + self._max_batch_size]))
        except Exception as error:  # noqa: BLE001 - never leave a waiter blocked
            self._flushing = False
            for text in tuple(self._slots):
                slot = self._slots.pop(text)
                if not slot.event.is_set():
                    slot.error = error
                    slot.event.set()

    async def _run_batch(self, texts):
        try:
            result = await self._model.embed(texts)
            vectors = tuple(result.vectors)
            if len(vectors) != len(texts):
                raise ValueError("embedding vector count does not match the batch")
        except Exception as error:  # noqa: BLE001 - the failure belongs to every waiter
            for text in texts:
                slot = self._slots.pop(text, None)
                if slot is not None:
                    slot.error = error
                    slot.event.set()
            return

        if self._cache_size > 0:
            for text, vector in zip(texts, vectors):
                self._cache[text] = vector
                self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        for text, vector in zip(texts, vectors):
            slot = self._slots.pop(text, None)
            if slot is not None:
                slot.vector = vector
                slot.usage = result.usage
                slot.event.set()


def _sum_usage(batches):
    requests = 0
    input_tokens = None
    output_tokens = None
    for usage in batches:
        requests += usage.requests
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
