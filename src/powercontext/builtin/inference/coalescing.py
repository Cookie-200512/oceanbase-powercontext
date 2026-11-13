"""Cross-call embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from powercontext.builtin.inference import EmbeddingResult, InferenceUsage

_DEFAULT_MAX_BATCH_SIZE = 64
_DEFAULT_COALESCE_MS = 5.0
_DEFAULT_CACHE_SIZE = 1024


class _Batch:
    """One underlying model request and the texts it covers."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts):
        self.texts = tuple(texts)
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
        self._futures = {}
        self._pending = []
        self._flushing = False

    async def embed(self, texts):
        """Embed texts in order, sharing duplicate and concurrent work."""
        items = tuple(texts)
        if not items:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(items))
        vectors = {}
        waiting = []
        start_flush = False

        for text in unique:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            future = self._futures.get(text)
            if future is None:
                future = asyncio.get_event_loop().create_future()
                self._futures[text] = future
                self._pending.append(text)
                if not self._flushing:
                    self._flushing = True
                    start_flush = True
            waiting.append(text)

        if start_flush:
            asyncio.ensure_future(self._flush_later())

        if waiting:
            gathered = await asyncio.gather(
                *(asyncio.shield(self._futures[text]) for text in waiting)
            )
            batches = {}
            for text, outcome in zip(waiting, gathered):
                vector, batch = outcome
                vectors[text] = vector
                batches[id(batch)] = batch
            usage = _sum_usage(tuple(batches.values()))
        else:
            usage = InferenceUsage(requests=0)

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in items),
            usage=usage,
        )

    async def _flush_later(self):
        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)
        pending = self._pending
        self._pending = []
        self._flushing = False
        for start in range(0, len(pending), self._max_batch_size):
            batch = _Batch(pending[start : start + self._max_batch_size])
            asyncio.ensure_future(self._execute(batch))

    async def _execute(self, batch):
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:
            for text in batch.texts:
                future = self._futures.pop(text, None)
                if future is not None and not future.done():
                    future.cancel()
            raise
        except Exception as error:
            for text in batch.texts:
                future = self._futures.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)
            return

        vectors = tuple(result.vectors)
        if len(vectors) != len(batch.texts):
            error = ValueError("embedding vector count does not match the batch")
            for text in batch.texts:
                future = self._futures.pop(text, None)
                if future is not None and not future.done():
                    future.set_exception(error)
            return

        batch.usage = result.usage
        if self._cache_size > 0:
            for text, vector in zip(batch.texts, vectors):
                self._cache[text] = vector
                self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        for text, vector in zip(batch.texts, vectors):
            future = self._futures.pop(text, None)
            if future is not None and not future.done():
                future.set_result((vector, batch))


def _sum_usage(batches):
    requests = 0
    input_tokens = None
    output_tokens = None
    for batch in batches:
        usage = batch.usage
        if usage is None:
            continue
        requests += usage.requests
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
