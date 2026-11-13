"""HTTP cross-encoder reranking for coarse Memory search candidates."""

from __future__ import annotations

import math
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _RerankCandidate(_StrictModel):
    rank: int = Field(ge=1)
    text: str


class _RerankRequest(_StrictModel):
    query: str
    candidates: tuple[_RerankCandidate, ...]


class _RerankScore(_StrictModel):
    rank: int = Field(ge=1)
    score: float


class _RerankResponse(_StrictModel):
    scores: tuple[_RerankScore, ...]


class HTTPMemoryReranker:
    """Score Memory candidates with one or more bounded HTTP batch requests."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = DEFAULT_HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must not be blank")  # noqa: TRY003
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")  # noqa: TRY003
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must not be blank")  # noqa: TRY003
        self.endpoint = endpoint
        self.policy_id = policy_id
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate once, then select globally across batches."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")  # noqa: TRY003
        if not candidates:
            raise ValueError("rerank requires at least one candidate")  # noqa: TRY003
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError("limit must be an integer")  # noqa: TRY003
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")  # noqa: TRY003

        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            ranks = tuple(range(start + 1, start + 1 + len(batch)))
            payload = _RerankRequest(
                query=query,
                candidates=tuple(
                    _RerankCandidate(rank=rank, text=hit.text)
                    for rank, hit in zip(ranks, batch, strict=True)
                ),
            ).model_dump(mode="json")
            response = await self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
            requests += 1
            for rank, score in self._parse_scores(response, ranks):
                scores[rank] = score

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )

    @staticmethod
    def _parse_scores(response: httpx.Response, expected: tuple[int, ...]) -> list[tuple[int, float]]:
        """Validate one response body and return its rank/score pairs."""

        try:
            payload: Any = response.json()
        except ValueError as error:
            raise ValueError("rerank response is not valid JSON") from error
        try:
            parsed = _RerankResponse.model_validate(payload)
        except ValidationError as error:
            raise ValueError("rerank response does not match the score schema") from error

        observed = [item.rank for item in parsed.scores]
        if len(observed) != len(expected) or set(observed) != set(expected):
            raise ValueError("rerank response must cover each batch rank exactly once")  # noqa: TRY003

        pairs: list[tuple[int, float]] = []
        for item in parsed.scores:
            if not math.isfinite(item.score):
                raise ValueError("rerank score must be finite")  # noqa: TRY003
            pairs.append((item.rank, item.score))
        return pairs


__all__ = [
    "DEFAULT_HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
]
