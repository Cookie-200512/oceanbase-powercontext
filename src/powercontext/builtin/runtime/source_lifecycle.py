"""Source-dependent lifecycle for Memory and Topic Memory availability."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation

LIFECYCLE_SCHEMA = "powercontext.source-lifecycle.v1"
LIFECYCLE_FAMILY = "source-lifecycle"
_PROCESS_STATE: dict[str, dict[str, Any]] = {}
_POLICIES = ("all", "any")


class SourceLifecycleError(ValueError):
    """A lifecycle graph, event or reference violated the contract."""


class SourceLifecycleEvent(BaseModel):
    """One validated availability change for a registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    reason: str
    revision: int = Field(strict=True, gt=0)
    sequence: int = Field(strict=True, gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")  # noqa: TRY003
        return value


def _require_node_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceLifecycleError("node_id must not be blank")  # noqa: TRY003
    return value


class ScopedSourceLifecycleApplication:
    """Register one scope's evidence DAG and propagate Source availability."""

    def __init__(self, runtime: Any, scope_id: str) -> None:
        self._runtime = runtime
        self.scope_id = scope_id
        self._state: dict[str, Any] | None = None

    # ------------------------------------------------------------------ graph
    async def register_source(self, node_id: str, source_ref: Any, *, revision: int) -> None:
        node_id = _require_node_id(node_id)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise SourceLifecycleError("revision must be a positive integer")  # noqa: TRY003
        await self._runtime.records.for_scope(self.scope_id).get_source(
            source_ref.source_type, source_ref.source_id
        )
        node = {
            "kind": "source",
            "ref": source_ref.model_dump(mode="json"),
            "parents": [],
            "policy": "all",
            "revision": revision,
            "sequence": 0,
            "available": True,
        }
        await self._register(node_id, node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: str = "all",
    ) -> None:
        node_id = _require_node_id(node_id)
        from powercontext.builtin.runtime.models import GetMemoryEntryRequest

        await self._runtime.memory.for_scope(self.scope_id).get(
            GetMemoryEntryRequest(citation=citation)
        )
        node = {
            "kind": "memory",
            "ref": citation.model_dump(mode="json"),
            "parents": list(parents),
            "policy": _require_policy(policy),
            "revision": 1,
            "sequence": 0,
            "available": True,
        }
        await self._register(node_id, node)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: str = "all",
    ) -> None:
        node_id = _require_node_id(node_id)
        await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
            artifact_ref.family, artifact_ref.artifact_id, artifact_ref.revision
        )
        node = {
            "kind": "topic",
            "ref": artifact_ref.model_dump(mode="json"),
            "parents": list(parents),
            "policy": _require_policy(policy),
            "revision": 1,
            "sequence": 0,
            "available": True,
        }
        await self._register(node_id, node)

    async def _register(self, node_id: str, node: dict[str, Any]) -> None:
        state = await self._load()
        nodes = state["nodes"]
        existing = nodes.get(node_id)
        if existing is not None:
            if existing != node:
                raise SourceLifecycleError(f"node {node_id!r} is already registered")  # noqa: TRY003
            return
        parents = node["parents"]
        if len(set(parents)) != len(parents):
            raise SourceLifecycleError("duplicate parent")  # noqa: TRY003
        if node_id in parents:
            raise SourceLifecycleError("self reference")  # noqa: TRY003
        if node["kind"] != "source" and not parents:
            raise SourceLifecycleError("derived nodes require a parent")  # noqa: TRY003
        for parent in parents:
            if parent not in nodes:
                raise SourceLifecycleError(f"unknown parent {parent!r}")  # noqa: TRY003
        node["available"] = _derive(node, nodes)
        nodes[node_id] = node
        await self._sync_memory(nodes)
        await self._save(state)

    # ----------------------------------------------------------------- events
    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        state = await self._load()
        nodes = state["nodes"]
        node = nodes.get(event.node_id)
        if node is None or node["kind"] != "source":
            raise SourceLifecycleError("event requires a registered Source")  # noqa: TRY003
        if node["revision"] != event.revision:
            raise SourceLifecycleError("lifecycle revision mismatch")  # noqa: TRY003
        previous_sequence = node["sequence"]
        if event.sequence < previous_sequence:
            raise SourceLifecycleError("sequence must increase")  # noqa: TRY003
        if event.sequence == previous_sequence:
            if node["available"] != event.available:
                raise SourceLifecycleError("sequence replay conflicts with stored state")  # noqa: TRY003
            return sorted(name for name, item in nodes.items() if item["available"])
        node["sequence"] = event.sequence
        node["available"] = event.available

        for name, item in nodes.items():
            item["available"] = _derive(item, nodes)

        await self._sync_memory(nodes)
        await self._save(state)
        return sorted(name for name, item in nodes.items() if item["available"])

    async def _sync_memory(self, nodes: dict[str, Any]) -> None:
        from powercontext.builtin.runtime.models import (
            GetMemoryEntryRequest,
            RetireMemoryEntryRequest,
        )

        memory = self._runtime.memory.for_scope(self.scope_id)
        for item in nodes.values():
            if item['kind'] != 'memory':
                continue
            citation = MemoryCitation.model_validate(item['ref'])
            # The Memory head moves with every state change, so resolve the
            # current citation for this entry right before mutating it.
            snapshot = await memory.list(include_inactive=True)
            record = next(
                (entry for entry in snapshot.entries if entry.entry.entry_id == citation.entry_id),
                None,
            )
            if record is None:
                raise SourceLifecycleError('registered Memory entry is missing')
            active = record.state == 'active'
            if item['available'] and not active:
                await memory.restore(GetMemoryEntryRequest(citation=record.citation))
            elif not item['available'] and active:
                await memory.retire(
                    RetireMemoryEntryRequest(citation=record.citation, reason='source lifecycle')
                )

    # ---------------------------------------------------------------- resolve
    async def resolve(self, node_id: str, *, purpose: str) -> Any:
        state = await self._load()
        node = state["nodes"].get(node_id)
        if node is None:
            raise SourceLifecycleError(f"unknown node {node_id!r}")  # noqa: TRY003
        if purpose not in ("generation", "audit"):
            raise SourceLifecycleError("unknown purpose")  # noqa: TRY003
        if purpose == "generation" and not node["available"]:
            raise SourceLifecycleError(f"node {node_id!r} is unavailable")  # noqa: TRY003
        return _decode_ref(node)

    # ------------------------------------------------------------ persistence
    async def _load(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        state = _PROCESS_STATE.get(self.scope_id)
        if state is None:
            state = {"schema": LIFECYCLE_SCHEMA, "nodes": {}}
            _PROCESS_STATE[self.scope_id] = state
        self._state = state
        return state

    async def _save(self, state: dict[str, Any]) -> None:
        _PROCESS_STATE[self.scope_id] = state


def _require_policy(policy: str) -> str:
    if policy not in _POLICIES:
        raise SourceLifecycleError("policy must be 'all' or 'any'")  # noqa: TRY003
    return policy


def _derive(node: dict[str, Any], nodes: dict[str, Any]) -> bool:
    if node["kind"] == "source":
        return bool(node["available"])
    support = [bool(nodes[parent]["available"]) for parent in node["parents"]]
    return all(support) if node["policy"] == "all" else any(support)


def _decode_ref(node: dict[str, Any]) -> Any:
    from powercontext.sources.models import SourceRef

    kind = node["kind"]
    if kind == "source":
        return SourceRef.model_validate(node["ref"])
    if kind == "memory":
        return MemoryCitation.model_validate(node["ref"])
    return ArtifactRef.model_validate(node["ref"])


class SourceLifecycleApplication:
    """Select the scoped lifecycle service for one Runtime."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycleApplication:
        return ScopedSourceLifecycleApplication(self._runtime, scope_id)


__all__ = [
    "LIFECYCLE_FAMILY",
    "LIFECYCLE_SCHEMA",
    "SourceLifecycleApplication",
    "SourceLifecycleError",
    "SourceLifecycleEvent",
    "ScopedSourceLifecycleApplication",
]


def filter_topic_result(scope_id: str, result: Any) -> Any:
    """Drop Topic hits whose registered artifact is not currently available."""
    state = _PROCESS_STATE.get(scope_id)
    if not state:
        return result
    inactive = {
        node['ref'].get('artifact_id')
        for node in state['nodes'].values()
        if node['kind'] == 'topic' and not node['available']
    }
    if not inactive:
        return result
    hits = tuple(hit for hit in result.hits if hit.artifact_ref.artifact_id not in inactive)
    return result.model_copy(update={'hits': hits})
