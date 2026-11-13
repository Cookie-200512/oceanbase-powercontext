"""Context Pack wire format, deterministic chunking and prefix deltas."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from typing import Any

CONTEXT_PACK_SCHEMA = "powercontext.context-pack.v1"
CONTEXT_PACK_DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512

_PACK_FIELDS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(ValueError):
    """A wire payload, digest or chunking rule was violated."""


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _digest_for(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": CONTEXT_PACK_SCHEMA, "items": _encode_items(items)}
    return "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContextPackError(f"{name} must be a sha256 digest")  # noqa: TRY003
    if any(char not in "0123456789abcdef" for char in value[7:]):
        raise ContextPackError(f"{name} must be a lowercase hex digest")  # noqa: TRY003
    return value


def _decode_items(value: Any, name: str) -> tuple[bytes, ...]:
    if not isinstance(value, list):
        raise ContextPackError(f"{name} must be a list")  # noqa: TRY003
    decoded: list[bytes] = []
    for entry in value:
        if not isinstance(entry, str):
            raise ContextPackError(f"{name} entries must be base64 strings")  # noqa: TRY003
        try:
            decoded.append(base64.b64decode(entry.encode("ascii"), validate=True))
        except (UnicodeEncodeError, binascii.Error) as error:
            raise ContextPackError(f"{name} contains invalid base64") from error  # noqa: TRY003
    return tuple(decoded)


def _load_object(data: bytes, fields: frozenset[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(f"{label} is not valid UTF-8 JSON") from error  # noqa: TRY003
    if not isinstance(payload, dict):
        raise ContextPackError(f"{label} must be a JSON object")  # noqa: TRY003
    if set(payload) != fields:
        raise ContextPackError(f"{label} fields do not match the wire contract")  # noqa: TRY003
    return payload


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete context payload split into verified chunks."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        """Build a pack from in-memory chunks and compute its canonical size."""
        chunks = tuple(bytes(item) for item in items)
        digest = _digest_for(chunks)
        payload: dict[str, Any] = {
            "schema": CONTEXT_PACK_SCHEMA,
            "items": _encode_items(chunks),
            "digest": digest,
        }
        size = 0
        while True:
            payload["byte_count"] = size
            encoded = _canonical(payload)
            if len(encoded) == size:
                break
            size = len(encoded)
        return cls(items=chunks, digest=digest, byte_count=size)

    @classmethod
    def from_content(cls, content: str | bytes, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Split content into deterministic UTF-8 chunks within the byte limit."""
        if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")  # noqa: TRY003
        if isinstance(content, bytes):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ContextPackError("content must be valid UTF-8") from error  # noqa: TRY003
        elif isinstance(content, str):
            text = content
        else:
            raise ContextPackError("content must be str or bytes")  # noqa: TRY003

        chunks: list[bytes] = []
        current = bytearray()
        for char in text:
            encoded = char.encode("utf-8")
            if len(encoded) > chunk_bytes:
                raise ContextPackError("chunk_bytes cannot hold one character")  # noqa: TRY003
            if len(current) + len(encoded) > chunk_bytes:
                chunks.append(bytes(current))
                current = bytearray()
            current.extend(encoded)
        if current:
            chunks.append(bytes(current))
        return cls.build(tuple(chunks))

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and verify a complete pack payload."""
        payload = _load_object(data, _PACK_FIELDS, "context pack")
        if payload["schema"] != CONTEXT_PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")  # noqa: TRY003
        items = _decode_items(payload["items"], "items")
        digest = _require_digest(payload["digest"], "digest")
        byte_count = payload["byte_count"]
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise ContextPackError("byte_count must be a non-negative integer")  # noqa: TRY003
        if byte_count != len(data):
            raise ContextPackError("byte_count does not match the payload")  # noqa: TRY003
        if digest != _digest_for(items):
            raise ContextPackError("pack digest does not match its items")  # noqa: TRY003
        return cls(items=items, digest=digest, byte_count=byte_count)

    def to_bytes(self) -> bytes:
        """Serialize this pack to its canonical wire form."""
        return _canonical(
            {
                "schema": CONTEXT_PACK_SCHEMA,
                "items": _encode_items(self.items),
                "digest": self.digest,
                "byte_count": self.byte_count,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The retained base prefix and the appended target suffix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        """Keep the longest shared chunk prefix and append the target suffix."""
        prefix = 0
        for left, right in zip(base.items, target.items, strict=False):
            if left != right:
                break
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Actual wire size of this delta, including protocol fields."""
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta to its canonical wire form."""
        return _canonical(
            {
                "schema": CONTEXT_PACK_DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": _encode_items(self.appended),
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and verify a delta payload."""
        payload = _load_object(data, _DELTA_FIELDS, "context pack delta")
        if payload["schema"] != CONTEXT_PACK_DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")  # noqa: TRY003
        base_digest = _require_digest(payload["base_digest"], "base_digest")
        target_digest = _require_digest(payload["target_digest"], "target_digest")
        prefix_items = payload["prefix_items"]
        if not isinstance(prefix_items, int) or isinstance(prefix_items, bool) or prefix_items < 0:
            raise ContextPackError("prefix_items must be a non-negative integer")  # noqa: TRY003
        appended = _decode_items(payload["appended"], "appended")
        return cls(
            base_digest=base_digest,
            target_digest=target_digest,
            prefix_items=prefix_items,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Rebuild the target pack from a base pack and one delta."""
    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("apply requires a pack and a delta")  # noqa: TRY003
    if delta.base_digest != base.digest:
        raise ContextPackError("delta base digest does not match the base pack")  # noqa: TRY003
    prefix = delta.prefix_items
    if prefix < 0 or prefix > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack")  # noqa: TRY003
    rebuilt = ContextPack.build(base.items[:prefix] + delta.appended)
    if rebuilt.digest != delta.target_digest:
        raise ContextPackError("rebuilt pack digest does not match the delta target")  # noqa: TRY003
    return rebuilt


__all__ = [
    "CONTEXT_PACK_DELTA_SCHEMA",
    "CONTEXT_PACK_SCHEMA",
    "DEFAULT_CHUNK_BYTES",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
