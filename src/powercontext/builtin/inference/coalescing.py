"""Cross-call embedding request coalescing with a bounded LRU vector cache."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage


class _Batch:
    """One underlying model request and the texts it covers."""

    __slots__ = ("texts", "usage")

    def __init__(self, texts: tuple[str, ...]) -> None:
        self.texts = texts
        self.usage: InferenceUsage | None = None


class CoalescingEmbeddingModel:
    """Share in-flight embedding work and cache successful vectors between calls."""

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(coalesce_ms, (int, float)) or isinstance(coalesce_ms, bool) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a non-negative number")  # noqa: TRY003
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._futures: dict[str, asyncio.Future[tuple[tuple[float, ...], _Batch]]] = {}
        self._pending: list[str] = []
        self._flush_handle: asyncio.TimerHandle | None = None

    async def embed(self, texts: tuple[str, ...]) -> EmbeddingResult:
        """Embed texts in order, sharing duplicate and concurrent work."""
        items = tuple(texts)
        if not items:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = tuple(dict.fromkeys(items))
        vectors: dict[str, tuple[float, ...]] = {}
        waiting: list[str] = []

        for text in unique:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                vectors[text] = cached
                continue
            future = self._futures.get(text)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                self._futures[text] = future
                self._pending.append(text)
                self._schedule_flush()
            waiting.append(text)

        if waiting:
            outcomes = await asyncio.gather(
                *(asyncio.shield(self._futures[text]) for text in waiting)
            )
            batches: dict[int, _Batch] = {}
            for text, (vector, batch) in zip(waiting, outcomes, strict=True):
                vectors[text] = vector
                batches[id(batch)] = batch
            usage = _sum_usage(tuple(batches.values()))
        else:
            usage = InferenceUsage(requests=0)

        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in items),
            usage=usage,
        )

    def _schedule_flush(self) -> None:
        if self._flush_handle is not None:
            return
        loop = asyncio.get_running_loop()
        self._flush_handle = loop.call_later(self._coalesce_seconds, self._flush)

    def _flush(self) -> None:
        self._flush_handle = None
        pending = self._pending
        self._pending = []
        for start in range(0, len(pending), self._max_batch_size):
            batch = _Batch(tuple(pending[start : start + self._max_batch_size]))
            asyncio.get_running_loop().create_task(self._execute(batch))

    async def _execute(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:
            for text in batch.texts:
                future = self._futures.pop(text, None)
                if future is not None and not future.done():
                    future.cancel()
            raise
        except Exception as error:  # noqa: BLE001 - the failure belongs to every waiter
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
            for text, vector in zip(batch.texts, vectors, strict=True):
                self._cache[text] = vector
                self._cache.move_to_end(text)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        for text, vector in zip(batch.texts, vectors, strict=True):
            future = self._futures.pop(text, None)
            if future is not None and not future.done():
                future.set_result((vector, batch))


def _sum_usage(batches: tuple[_Batch, ...]) -> InferenceUsage:
    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
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
